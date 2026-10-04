"""In-process Cado recording around AReaL's RLVR trajectory, with strict failure handling."""

import os
import uuid
from pathlib import Path
from areal.workflow.rlvr import RLVRWorkflow


class AuditedRLVRWorkflow(RLVRWorkflow):
    def __init__(self, *, audit_on=False, audit_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.audit = None
        if audit_on:
            import tplane

            # No GPU resource cost is fabricated for overlapping per-trajectory records.
            gpu = tplane.no_gpu_reader()
            self.audit = tplane.Run(
                run_id=Path(audit_dir).parent.name,
                store=tplane.FileRecordStore(Path(audit_dir)),
                options=tplane.RunOptions(
                    policy=tplane.Policy(
                        on_agent=tplane.Action.ABORT,
                        on_grader=tplane.Action.ABORT,
                        on_infra=tplane.Action.ABORT,
                        on_timeout=tplane.Action.ABORT,
                        max_attempts=1,
                    )
                ),
                read_host=tplane.proc_status_reader(os.getpid()),
                read_gpu=gpu.read,
                gpu_total_bytes=gpu.total_bytes,
            )

    async def arun_episode(self, engine, data):
        if self.audit is None:
            return await super().arun_episode(engine, data)
        import tplane

        with self.audit.unit(
            uuid.uuid4().hex, kind=tplane.UnitKind.TRAJECTORY, source="areal"
        ) as unit:
            with unit.stage(tplane.Stage.ROLLOUT):
                result = await super().arun_episode(engine, data)
            prompt_len = int((result["loss_mask"] == 0).sum().item())
            output_len = int(result["loss_mask"].sum().item())
            unit.tokens(tokens_in=prompt_len, tokens_out=output_len)
            unit.reward(float(result["rewards"].item()))
            versions = result["versions"][result["loss_mask"].bool()]
            unit.tag("policy_head", str(int(versions.min().item())))
            unit.tag("policy_tail", str(int(versions.max().item())))
        # Exceptions abort this episode/run; never turn an infrastructure failure into reward=0.
        return result
