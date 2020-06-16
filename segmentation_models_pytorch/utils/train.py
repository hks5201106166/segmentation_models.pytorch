import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter
import numpy as np
import cv2
class Epoch:

    def __init__(self, model,loss_idcard_detection,loss_logo_detection,metrics_idcard_detection,metrics_logo_detection,stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss_idcard_detection = loss_idcard_detection
        self.loss_logo_detection=loss_logo_detection
        self.metrics_idcard_detection = metrics_idcard_detection
        self.metrics_logo_detection=metrics_logo_detection
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss_idcard_detection.to(self.device)
        self.loss_logo_detection.to(self.device)
        for metrics_idcard_detection in self.metrics_idcard_detection:
            metrics_idcard_detection.to(self.device)
        for metrics_logo_detection in self.metrics_logo_detection:
            metrics_logo_detection.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self,x,mask_idcard_detection,mask_logo_detection):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters_idcard_detection = {metric.__name__: AverageValueMeter() for metric in self.metrics_idcard_detection}
        metrics_meters_logo_detection = {metric.__name__: AverageValueMeter() for metric in
                                           self.metrics_logo_detection}

        iou_scores_idcard_detection = []
        iou_scores_logo_detection = []
        nums = 0
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for image,(mask_idcard_detection,mask_logo_detection),(gt_mask_idcard_detection,gt_mask_logo_detection) in iterator:
                image=image.to(self.device)
                mask_idcard_detection=mask_idcard_detection.to(self.device)
                mask_logo_detection=mask_logo_detection.to(self.device)

                gt_mask_idcard_detection=gt_mask_idcard_detection.to(self.device)
                gt_mask_logo_detection=gt_mask_logo_detection.to(self.device)

                loss, y_idcard_detection,y_logo_detection = self.batch_update(image,mask_idcard_detection,mask_logo_detection)
                #print(loss)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                #loss_logs = {self.loss.__name__: loss_meter.mean}
                loss_logs = {'loss': loss_meter.mean}
                logs.update(loss_logs)

                y_idcard_detection=torch.nn.Softmax2d()(y_idcard_detection)
                y_logo_detection=torch.nn.Softmax2d()(y_logo_detection)

                #logo_mask=logo_mask[0,0,:,:]
                # update metrics logs
                for metric_fn in self.metrics_idcard_detection:
                    #y_pred=torch.argmax(y_pred,dim=1)
                    metric_value = metric_fn(y_idcard_detection, gt_mask_idcard_detection).cpu().detach().numpy()
                    metrics_meters_idcard_detection[metric_fn.__name__].add(metric_value)
                for metric_fn in self.metrics_logo_detection:
                    #y_pred=torch.argmax(y_pred,dim=1)
                    metric_value = metric_fn(y_logo_detection, gt_mask_logo_detection).cpu().detach().numpy()
                    metrics_meters_logo_detection[metric_fn.__name__].add(metric_value)
                metrics_logs_idcard_detection = {k: v.mean for k, v in metrics_meters_idcard_detection.items()}
                metrics_logs_logo_detection = {k: v.mean for k, v in metrics_meters_logo_detection.items()}
                # logs.update(metrics_logs_idcard_detection)
                # logs.update(metrics_logs_logo_detection)
                iou_scores_idcard_detection.append(metrics_logs_idcard_detection['iou_score'])
                iou_scores_logo_detection.append(metrics_logs_logo_detection['iou_score'])
                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
                nums += 1
        print('average iou_scors_idcard_detection:{}'.format(sum(iou_scores_idcard_detection) / float(nums)))
        print('average iou_scors_logo_detection:{}'.format(sum(iou_scores_logo_detection) / float(nums)))

        return logs


class TrainEpoch(Epoch):

    def __init__(self,model, loss_idcard_detection,loss_logo_detection,metrics_idcard_detection,metrics_logo_detection, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss_idcard_detection=loss_idcard_detection,
            loss_logo_detection=loss_logo_detection,
            metrics_idcard_detection=metrics_idcard_detection,
            metrics_logo_detection=metrics_logo_detection,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self,x, mask_idcard_detection,mask_logo_detection,stage='train'):


        self.optimizer.zero_grad()
        prediction_idcard_detection, prediction_logo_detection = self.model.forward(x)
        loss_idcard_detection = self.loss_idcard_detection(prediction_idcard_detection,
                                                           mask_idcard_detection.long())
        loss_logo_detection = self.loss_logo_detection(prediction_logo_detection, mask_logo_detection)
        loss = loss_logo_detection + loss_idcard_detection
        loss_idcard_detection = self.loss_idcard_detection(prediction_idcard_detection, mask_idcard_detection.long())
        loss_logo_detection=self.loss_logo_detection(prediction_logo_detection,mask_logo_detection)
        loss=loss_logo_detection+loss_idcard_detection
        loss.backward()
        self.optimizer.step()
        return loss, prediction_idcard_detection,prediction_logo_detection


class ValidEpoch(Epoch):

    def __init__(self, model, loss_idcard_detection,loss_logo_detection,metrics_idcard_detection,metrics_logo_detection, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss_idcard_detection=loss_idcard_detection,
            loss_logo_detection=loss_logo_detection,
            metrics_idcard_detection=metrics_idcard_detection,
            metrics_logo_detection=metrics_logo_detection,
            stage_name='valid',
            device=device,
            verbose=verbose,)

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, mask_idcard_detection,mask_logo_detection):
        with torch.no_grad():
            prediction_idcard_detection,prediction_logo_detection = self.model.forward(x)
            loss_idcard_detection = self.loss_idcard_detection(prediction_idcard_detection,
                                                               mask_idcard_detection.long())
            loss_logo_detection = self.loss_logo_detection(prediction_logo_detection, mask_logo_detection)
            loss = loss_logo_detection + loss_idcard_detection
        return loss, prediction_idcard_detection,prediction_logo_detection
