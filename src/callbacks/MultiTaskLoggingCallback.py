import logging

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class MultiTaskLoggingCallback(TrainerCallback):
    """
    Callback to log individual task losses
    """

    def __init__(self):
        super().__init__()
        self.mlm_losses = []
        self.scp_losses = []
        self.acm_losses = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        This is called after each training step
        Unfortunately, we don't have direct access to outputs here,
        so we need the custom Trainer approach above for proper logging.
        """
        pass

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        """
        Called when logging occurs
        """
        if logs is not None:
            # The custom trainer will have already added these
            if 'train/mlm_loss' in logs:
                logger.info(f"\n📊 Step {state.global_step} Losses:")
                logger.info(f"   MLM Loss: {logs.get('train/mlm_loss', 'N/A'):.4f}")
                logger.info(f"   SCP Loss: {logs.get('train/scp_loss', 'N/A'):.4f}")
                logger.info(f"   ACM Loss: {logs.get('train/acm_loss', 'N/A'):.4f}")
                logger.info(f"   Total Loss: {logs.get('train/total_loss', logs.get('loss', 'N/A')):.4f}")
