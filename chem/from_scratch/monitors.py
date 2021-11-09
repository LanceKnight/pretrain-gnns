from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class MetricMonitor(Callback):
    def __init__(self, stage='train', metric=None, logger=None, logging_interval=None, title=None, series=None):

        if logging_interval not in (None, "step", "epoch"):
            raise MisconfigurationException("MetricMonitor: logging_interval should be `step` or `epoch` or `None`.")
        if metric is None:
            raise MisconfigurationException("MetricMonitor: metric is not specified")
        if stage not in ('both', 'train', 'valid'):
            raise MisconfigurationException(f"MetricMonitor: input 'stage' argument = {stage}, which cannot be recognized")
        self.logger = logger
        self.metric = metric
        self.logging_interval = logging_interval
        self.stage = stage
        self.title = title
        self.series = series

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if 'train' in self.stage:
            if self.logging_interval == "step":
                series = self.series if self.series is not None else 'train'
                print(f'on_train_batch_end:title={self.title}, series={series}, value={outputs[self.metric]}, trainer.global_step={trainer.global_step}')
                self.logger.report_scalar(title=self.title, series=series, value=outputs[self.metric], iteration=trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        if 'train' in self.stage:
            if self.logging_interval == "epoch":
                outputs = pl_module.train_epoch_outputs
                series = self.series if self.series is not None else 'train'
                print(f'on_train_epoch_end: title={self.title}, series={series}, value={outputs[self.metric]}, trainer.current_epoch={trainer.current_epoch}')
                self.logger.report_scalar(title=self.title, series=series, value=outputs[self.metric], iteration=trainer.current_epoch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if 'valid' in self.stage:
            if self.logging_interval == "step":
                series = self.series if self.series is not None else 'valid'
                print(f'on_validation_batch_end title={self.title}, series={series}, value={outputs[self.metric]}, trainer.global_step={trainer.global_step}')
                self.logger.report_scalar(title=self.title, series=series, value=outputs[self.metric], iteration=trainer.global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        if 'valid' in self.stage:
            if self.logging_interval == "epoch":
                outputs = pl_module.valid_epoch_outputs
                series = self.series if self.series is not None else 'valid'
                print(f'on_validation_epoch_end title={self.title}, series={series}, value={outputs[self.metric]}, trainer.current_epoch={trainer.current_epoch}')
                self.logger.report_scalar(title=self.title, series=series, value=outputs[self.metric], iteration=trainer.current_epoch)

    # special
    # def on_validation_epoch_end(self, trainer, pl_module):
    #     if 'valid' in self.stage:
    #         if self.logging_interval == "epoch":
    #             outputs = pl_module.valid_epoch_outputs
    #             series = self.series if self.series is not None else 'valid'
    #             print(f'on_validation_epoch_end: title={self.title}  series={series} value:{outputs[self.metric]}, trainer.global_step={trainer.global_step} ')
    #             self.logger.report_scalar(title=self.title, series=series, value=outputs[self.metric], iteration=trainer.global_step)

#=== loss


class LossMonitor(MetricMonitor):
    def __init__(self, stage='train', logger=None, logging_interval=None, title=None):
        super(LossMonitor, self).__init__(stage=stage, metric="loss", logger=logger, logging_interval=logging_interval, title=f'loss_by_{logging_interval}')


class LossNoDropoutMonitor(MetricMonitor):
    def __init__(self, stage='train', logger=None, logging_interval=None, title=None):
        super(LossNoDropoutMonitor, self).__init__(stage=stage, metric="loss_no_dropout", logger=logger, logging_interval=logging_interval, title=f'loss_by_{logging_interval}', series='no_dropout')

#=== logAUC


class LogAUCMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger=None, logging_interval=None, title=None):
        super(LogAUCMonitor, self).__init__(stage=stage, metric="logAUC", logger=logger, logging_interval=logging_interval, title=f'logAUC_by_{logging_interval}')


class LogAUCNoDropoutMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger=None, logging_interval=None, title=None):
        super(LogAUCNoDropoutMonitor, self).__init__(stage=stage, metric="logAUC_no_dropout", logger=logger,
                                                     logging_interval=logging_interval, title=f'logAUC_by_{logging_interval}', series='no_dropout')

#=== ppv


class PPVMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger=None, logging_interval=None, title=None):
        super(PPVMonitor, self).__init__(stage=stage, metric="ppv", logger=logger, logging_interval=logging_interval, title=f'PPV_by_{logging_interval}')


class PPVNoDropoutMonitor(MetricMonitor):
    def __init__(self, stage='valid', logger=None, logging_interval=None, title=None):
        super(PPVNoDropoutMonitor, self).__init__(stage=stage, metric="ppv_no_dropout", logger=logger, logging_interval=logging_interval, title=f'PPV_by_{logging_interval}', series='no_dropout')
