import numpy as np
import torch.nn.functional as F
import torch
from collections import OrderedDict


class Inference_Wuzhangai(object):
    def __init__(self, params, annotations = None):
        self.params = params
        self.annotations = annotations

        self.num_classes = params['num_classes']
        self.mode = params['mode']

        self.solved_instances_annotations = {}
        self.solved_instances_id = []
        self.tobesolved_instances_annotations = {}
        self.tobesolved_instances_id = []
        self.old_workers = []

        self.init_infered_posterior = []
        self.all_instances_posterior = []
        self.all_workers_ability = None
        self.infered_posterior_old = None

        self.current_all_workers = set()
        self.worker_old_normalizer_all, self.worker_old_pi_all = None, None

        self.sample_classes_init = {}

        self.consider_prior = params['consider_prior']
        self.prior = []




    def solve_kl(self, infered_posterior, truth):
        kl_mean = 0.0
        for j in range(len(infered_posterior)):
            infered_posterior_agent = []
            infered_posterior_agent_task = []
            for key in infered_posterior[j]:
                infered_posterior_agent_task.append(infered_posterior[j][key])

            infered_posterior_agent.append(np.array(infered_posterior_agent_task))
            infered_posterior_agent = np.array(infered_posterior_agent)

            truth_agent = []
            truth_task = []
            for key in truth[j]:
                truth_task.append(truth[j][key])

            truth_agent.append(np.array(truth_task))
            truth_agent = np.array(truth_agent)

            infered_posterior_agent[infered_posterior_agent == 0] = 0.00001
            truth_agent[truth_agent == 0] = 0.00001
            kl_mean += F.kl_div(torch.tensor(np.log(infered_posterior_agent)), torch.tensor(truth_agent),
                               reduction='mean')

        return kl_mean



    def test(self, infered_posterior, truth):
        acc_all_task = []

        infered_posterior_perpare = []
        for j in range(len(infered_posterior)):
            sent = []
            for _, value in infered_posterior[j].items():
                sent.append(value)
            infered_posterior_perpare.append(np.array(sent))
        infered_posterior_perpare = np.array(infered_posterior_perpare)

        for j in range(len(infered_posterior_perpare)):
            pred = np.argmax(infered_posterior_perpare[j], axis=1)
            acc = sum([1 if p == y else 0 for p, y in zip(pred, truth[:, j])]) / len(pred)
            acc_all_task.append(acc)

        return acc_all_task



    def inference_init(self, answers=None):

        instance_num = len(answers)
        # print("instance_num",instance_num)
        # print("answers",answers)
        # exit()

        # Iterate tasks
        pred_all_task = []

        if self.params["label_hard"] == "True":
            for j in range(len(self.num_classes)):
                pred = 1
                adjustment_factor = np.zeros((instance_num, self.num_classes[j]))
                i = 0
                for key, value in answers.items():
                    for key_this in value[j].keys():
                        if value[j][key_this] != -1:
                            adjustment_factor[i, value[j][key_this]] += 1
                    i += 1
                pred = adjustment_factor * pred
                pred = pred / np.sum(pred, 1).reshape(pred.shape[0], 1)
                pred_all_task.append(pred)

        if self.params["label_hard"] == "False":
            for j in range(len(self.num_classes)):
                pred = 1
                adjustment_factor = np.zeros((instance_num, self.num_classes[j]))
                i = 0
                for key, value in answers.items():
                    for key_this in value[j].keys():
                        if value[j][key_this] != -1:
                            if value[j][key_this] % 1 == 0:
                                adjustment_factor[i, int(value[j][key_this])] += 1
                            else:
                                position_first = int(value[j][key_this] // 1)
                                position_after = position_first + 1
                                decimal_part = value[j][key_this] % 1
                                adjustment_factor[i, position_first] += (1-decimal_part)
                                adjustment_factor[i, position_after] += decimal_part
                    i += 1
                pred = adjustment_factor * pred
                pred = pred / np.sum(pred, 1).reshape(pred.shape[0], 1)
                pred_all_task.append(pred)

        for j in range(len(self.num_classes)):
            init_infered_posterior_task = {}
            i = 0
            for key, value in answers.items():
                #  key 指的是样本点的名称
                init_infered_posterior_task[key] = pred_all_task[j][i]
                i += 1
            self.init_infered_posterior.append(init_infered_posterior_task)

        # print("-----------------------************************")
        # print("", type(self.num_classes))
        # print("self.num_classes[0]",self.num_classes[0])
        # print("len(self.num_classes)",len(self.num_classes))
        # print("answers.items():", answers.items())
        # print("answers.items():",answers.items())
        # print("self.init_infered_posterior",self.init_infered_posterior)
        # exit()

    def e_step(self, answers, worker_ability):
        self.all_instances_posterior = []
        instance_num, worker_num = len(answers), len(self.current_all_workers)
        pred_all_task = []
        if self.params["label_hard"] == "True":
            for j in range(len(self.num_classes)):
                pred = 1
                if self.consider_prior == "False":
                    adjustment_factor = 100000 * np.ones((instance_num, self.num_classes[j]))
                else:
                    adjustment_factor = 100000 * np.tile(self.prior, (instance_num, 1))
                i = 0
                for _, value in answers.items():
                    for w in self.current_all_workers:
                        if value[j][w] != -1:
                            adjustment_factor[i] *= worker_ability[j][w][:, value[j][w]]
                    i += 1
                pred = adjustment_factor * pred
                pred = pred / np.sum(pred, 1).reshape(pred.shape[0], 1)
                pred_all_task.append(pred)

            for j in range(len(self.num_classes)):
                all_instances_posterior_task = {}
                i = 0
                for key, value in answers.items():
                    all_instances_posterior_task[key] = pred_all_task[j][i]
                    i += 1
                self.all_instances_posterior.append(all_instances_posterior_task)

        if self.params["label_hard"] == "False":
            for j in range(len(self.num_classes)):
                pred = 1
                if self.consider_prior == "False":
                    adjustment_factor = 100000 * np.ones((instance_num, self.num_classes[j]))
                else:
                    adjustment_factor = 100000 * np.tile(self.prior, (instance_num, 1))
                i = 0
                for _, value in answers.items():
                    for w in self.current_all_workers:
                        if value[j][w] != -1:
                            if value[j][w] % 1 == 0:
                                adjustment_factor[i] *= worker_ability[j][w][:, int(value[j][w])]
                            else:
                                position_first = int(value[j][w] // 1)
                                position_after = position_first + 1
                                decimal_part = value[j][w] % 1
                                decimal_part_2 = (1 - decimal_part)
                                pro_1 = worker_ability[j][w][:, position_first] * decimal_part_2
                                pro_2 = worker_ability[j][w][:, position_after] * decimal_part
                                adjustment_factor[i] *= (pro_1 + pro_2)
                    i += 1
                pred = adjustment_factor * pred
                pred = pred / np.sum(pred, 1).reshape(pred.shape[0], 1)
                pred_all_task.append(pred)

            for j in range(len(self.num_classes)):
                all_instances_posterior_task = {}
                i = 0
                for key, value in answers.items():
                    all_instances_posterior_task[key] = pred_all_task[j][i]
                    i += 1
                self.all_instances_posterior.append(all_instances_posterior_task)
            # print("self.all_instances_posterior",self.all_instances_posterior)

    def m_step(self, truth, annotations):
        # print("self.mode",self.mode)
        pi_all_task = []
        tmp = []
        if self.params["label_hard"] == "True":
            for j in range(len(self.num_classes)):
                pi = {}
                t = {}
                for w in self.current_all_workers:
                    pi[w] = np.zeros((self.num_classes[j], self.num_classes[j]))
                    normalizer = np.zeros(self.num_classes[j])
                    for id_key in truth[j].keys():
                        if annotations[id_key][j][w] != -1:
                            normalizer += truth[j][id_key]
                            pi[w][:, annotations[id_key][j][w]] += truth[j][id_key]
                    normalizer[normalizer == 0] = 0.00001
                    t[w] = pi[w]
                    pi[w] = pi[w] / normalizer.reshape(self.num_classes[j], 1)
                pi_all_task.append(pi)
                tmp.append(t)
        if self.params["label_hard"] == "False":
            for j in range(len(self.num_classes)):
                pi = {}
                t = {}
                for w in self.current_all_workers:
                    pi[w] = np.zeros((self.num_classes[j], self.num_classes[j]))
                    normalizer = np.zeros(self.num_classes[j])
                    for id_key in truth[j].keys():
                        if annotations[id_key][j][w] != -1:
                            normalizer += truth[j][id_key]
                            # pi[w][:, annotations[id_key][j][w]] += truth[j][id_key]
                            if annotations[id_key][j][w] % 1 == 0:
                                pi[w][:, int(annotations[id_key][j][w])] += truth[j][id_key]
                            else:
                                position_first = int(annotations[id_key][j][w] // 1)
                                position_after = position_first + 1
                                decimal_part = annotations[id_key][j][w] % 1
                                decimal_part_2 = (1 - decimal_part)
                                pi[w][:, position_first] += truth[j][id_key] * decimal_part_2
                                pi[w][:, position_after] += truth[j][id_key] * decimal_part
                    normalizer[normalizer == 0] = 0.00001
                    t[w] = pi[w]
                    pi[w] = pi[w] / normalizer.reshape(self.num_classes[j], 1)
                pi_all_task.append(pi)
                tmp.append(t)


        if self.consider_prior != "False":
            total_sum = np.zeros_like(next(iter(truth[0].values())))

            for key, distribution in truth[0].items():
                total_sum += distribution

            self.prior = total_sum / len(truth[0])

        self.all_workers_ability = pi_all_task
        # print("self.all_workers_ability", OrderedDict(sorted(self.all_workers_ability[0].items())))
        # print("self.all_workers_ability------", OrderedDict(sorted(tmp[0].items())))
        
        result = {}
        for rater, matrix in self.all_workers_ability[0].items():
            diag_sum = np.trace(matrix)
            result[rater] = diag_sum
        # print("workers trace:\n", result)
        # print()





    def m_step_save_information(self, truth, annotations):
        if self.params["label_hard"] == "True":
            normalizer_all = []
            pi_all = []
            for j in range(len(self.num_classes)):
                pi, normalizer = {}, {}
                for w in self.current_all_workers:
                    pi[w] = np.zeros((self.num_classes[j], self.num_classes[j]))
                    normalizer[w] = np.zeros(self.num_classes[j])
                    for id_key, id_value in annotations.items():
                        if id_value[j][w] != -1:
                            normalizer[w] += truth[j][id_key]
                            pi[w][:, annotations[id_key][j][w]] += truth[j][id_key]
                normalizer_all.append(normalizer)
                pi_all.append(pi)
        if self.params["label_hard"] == "False":
            normalizer_all = []
            pi_all = []
            for j in range(len(self.num_classes)):
                pi, normalizer = {}, {}
                for w in self.current_all_workers:
                    pi[w] = np.zeros((self.num_classes[j], self.num_classes[j]))
                    normalizer[w] = np.zeros(self.num_classes[j])
                    for id_key, id_value in annotations.items():
                        if id_value[j][w] != -1:
                            normalizer[w] += truth[j][id_key]
                            # pi[w][:, annotations[id_key][j][w]] += truth[j][id_key]
                            if annotations[id_key][j][w] % 1 == 0:
                                pi[w][:, int(annotations[id_key][j][w])] += truth[j][id_key]
                            else:
                                position_first = int(annotations[id_key][j][w] // 1)
                                position_after = position_first + 1
                                decimal_part = annotations[id_key][j][w] % 1
                                decimal_part_2 = (1 - decimal_part)
                                pi[w][:, position_first] += truth[j][id_key] * decimal_part_2
                                pi[w][:, position_after] += truth[j][id_key] * decimal_part

                normalizer_all.append(normalizer)
                pi_all.append(pi)


        return normalizer_all, pi_all





    def compute_current_all_workers(self):
        for key, value in self.annotations.items():
            for i, item in enumerate(value):
                for sent in item.keys():
                    self.current_all_workers.add(sent)
        print('self.current_all_workers', self.current_all_workers)
        print('-------------------------------------------------------------------------------------------')




    def calculate_class_difference_ratio(self, list1, list2):
        different_count = 0

        for sample1, sample2 in zip(list1, list2):
            for key in sample1:
                max_class_1 = np.argmax(sample1[key])
                max_class_2 = np.argmax(sample2[key])

                if max_class_1 != max_class_2:
                    different_count += 1

        ratio = different_count / len(list1)

        return ratio



    def train(self, worker_old_normalizer_all=None, worker_old_pi_all=None, truth=None, flag = False):
        self.worker_old_normalizer_all = worker_old_normalizer_all
        self.worker_old_pi_all = worker_old_pi_all

        self.compute_current_all_workers()

        # kl_divergence_divergence = [1.0] * self.params['patient']

        if flag != False:
            print('After initialization, the current ACC:', self.test(self.init_infered_posterior, truth))

        patient_class = 0

        for epoch in range(self.params['epoch']):
            print("epoch:", epoch+1)
            if epoch == 0:
                self.m_step(truth = self.init_infered_posterior, annotations = self.annotations)
            else:
                self.m_step(truth = self.all_instances_posterior, annotations = self.annotations)

            self.e_step(self.annotations, self.all_workers_ability)

            # result = {}
            # for rater, matrix in self.all_workers_ability[0].items():
            #     diag_sum = np.trace(matrix)
            #     result[rater] = diag_sum
            # print("workers trace:\n", OrderedDict(sorted(result.items())))

            # result = {}
            # for rater, matrix in self.all_workers_ability[0].items():
            #     diag_sum = matrix[0, 0] + matrix[-1, -1]
            #     result[rater] = diag_sum
            # print("workers first and last diagonal sum:\n", OrderedDict(sorted(result.items())))

            if flag != False:
                print('Epoch time:', epoch + 1, '   ACC of all tasks:', self.test(self.all_instances_posterior, truth))
            else:
                # print('Epoch time:', epoch + 1, '   infered posterior:', self.all_instances_posterior)
                print('Epoch time:', epoch + 1)
            print("-----------------------------------------------------")


            # # Determine whether the calculation is converged, and determine which instances' inference results are credible
            # if epoch != 0:
            #     # print("self.all_instances_posterior",self.all_instances_posterior)
            #     kl_divergence_divergence_new_instance = self.solve_kl(self.all_instances_posterior, self.infered_posterior_old)
            #     # print("kl_divergence_divergence_new_instance",kl_divergence_divergence_new_instance)
            #
            #     kl_divergence_divergence.pop(0)
            #     kl_divergence_divergence.append(kl_divergence_divergence_new_instance)
            #
            #     if min(kl_divergence_divergence) < self.params['kl_divergence_threshold']:
            #         print('\nOK, stop inference!')
            #         print("self.all_instances_posterior", self.all_instances_posterior)
            #         break

            if epoch != 0:
                if self.calculate_class_difference_ratio(self.infered_posterior_old, self.all_instances_posterior) < 0.02:
                    patient_class += 1
                else:
                    patient_class = 0

                if patient_class == self.params['patient']:
                    print('\nOK, stop inference!')
                    # print("self.all_instances_posterior", self.all_instances_posterior)
                    break


            self.infered_posterior_old = self.all_instances_posterior



        worker_normalizer_all, worker_pi_all = self.m_step_save_information(truth=self.all_instances_posterior, annotations=self.solved_instances_annotations)
        # print("worker_normalizer_all:", worker_normalizer_all)
        # print("worker_pi_all:", worker_pi_all)

        # final_result_path = "./results/" + "consider_prior_" + self.params['consider_prior'] + "/" + str(self.params['epoch'])
        # if not os.path.exists(final_result_path):
        #     os.makedirs(final_result_path)

        sample_classes = {}
        for sample_id, posterior in self.all_instances_posterior[0].items():
            class_index = np.argmax(posterior)
            sample_classes[sample_id] = class_index


        # np.save(os.path.join(final_result_path, 'all_instance_classes.npy'), sample_classes)
        # np.save(os.path.join(final_result_path, 'all_instance_classes_soft.npy'), self.all_instances_posterior[0])

        # np.save(os.path.join(final_result_path, 'worker_normalizer_all.npy'), worker_normalizer_all)
        # np.save(os.path.join(final_result_path, 'worker_pi_all.npy'), worker_pi_all)

        return self.all_instances_posterior, sample_classes, worker_normalizer_all, worker_pi_all