#使用BMIS_Metrics_Calculator 


metrics_list = ['Accuracy', 'DSC', 'Precision', 'Recall', 'IoU', 'F-Measure']
metrics_calculator = BMIS_Metrics_Calculator(metrics_list)

metrics_dict = metrics_calculator.get_metrics_dict(y_pred, y_true)

# 打印计算结果
for metric, value in metrics_dict.items():
    print(f"{metric}: {value}")

#示例：
class BMISegmentationExperiment(BaseSegmentationExperiment):
    def __init__(self, args):
        super(BMISegmentationExperiment, self).__init__(args)

        self.count = 1
        self.metrics_calculator = BMIS_Metrics_Calculator(args.metric_list)

    def inference(self):
        print("INFERENCE")
        self.model = load_model(self.args, self.model)
        test_results = self.inference_phase(self.args.final_epoch)

        return test_results

    def inference_phase(self, epoch):
        self.model.eval()

        total_loss, total = .0, 0
        total_metrics_dict = self.metrics_calculator.total_metrics_dict

        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(self.test_loader):
                if (batch_idx + 1) % self.args.step == 0:
                    print("EPOCH {} | {}/{}({}%) COMPLETE".format(epoch, batch_idx + 1, len(self.test_loader), np.round((batch_idx + 1) / len(self.test_loader) * 100), 4))

                loss, output, target = self.forward(image, target, mode='test')

                for idx, (target_, output_) in enumerate(zip(target, output)):
                    predict = torch.sigmoid(output_).squeeze()
                    metrics_dict = self.metrics_calculator.get_metrics_dict(predict, target_)

                    for metric in self.metrics_calculator.metrics_list:
                        total_metrics_dict[metric].append(metrics_dict[metric])

                total_loss += loss.item() * image.size(0)
                total += target.size(0)

        for metric in self.metrics_calculator.metrics_list:
            total_metrics_dict[metric] = np.round(np.mean(total_metrics_dict[metric]), 4)

        return total_metrics_dict
        
        
experiment = BMISegmentationExperiment(args)
test_results = experiment.inference()

print("Save MADGNet Test Results...")
save_metrics(args, test_results, model_dirs, args.final_epoch)

import os

def save_metrics(args, test_results, model_dirs, current_epoch):
    print("###################### TEST REPORT ######################")
    for metric in test_results.keys():
        print("Mean {}    :\t {}".format(metric, test_results[metric]))
    print("###################### TEST REPORT ######################\n")

    if args.train_data_type == args.test_data_type:
        test_results_save_path = os.path.join(model_dirs, 'test_reports', 'test_report(EPOCH {}).txt'.format(current_epoch))
    else:
        test_results_save_path = os.path.join(model_dirs, 'test_reports', 'Generalizability test_reports({}->{}).txt'.format(args.train_data_type, args.test_data_type))

    f = open(test_results_save_path, 'w')

    f.write("###################### TEST REPORT ######################\n")
    for metric in test_results.keys():
        f.write("Mean {}    :\t {}\n".format(metric, test_results[metric]))
    f.write("###################### TEST REPORT ######################\n")

    f.close()

    print("test results txt file is saved at {}".format(test_results_save_path))





#多分类
# 定义你要计算的指标列表
metrics_list = ['IoU', 'DSC', 'Recall']

# 创建指标计算器对象
metrics_calculator = Fundus_Image_Segmentation_Metrics_Calculator(metrics_list)

# 调用 get_metrics_dict 计算指标
metrics_dict = metrics_calculator.get_metrics_dict(y_pred, y_true)

# 输出结果
for metric, values in metrics_dict.items():
    print(f"Metric: {metric}")
    for label, value in values.items():
        print(f"  {label}: {value:.4f}")
        
        
TP_list, FP_list, TN_list, FN_list = MultiClassSegmentationMetrics(y_true, y_pred, num_classes=2)
