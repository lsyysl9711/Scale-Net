import os, re, time
from torch.optim import Adam as Optimizer

import envs
from utils import *

data_dict_cvrp = {
    "X-n101-k25": 27591,
    "X-n106-k14": 26362,
    "X-n110-k13": 14971,
    "X-n115-k10": 12747,
    "X-n120-k6": 13332,
    "X-n125-k30": 55539,
    "X-n129-k18": 28940,
    "X-n134-k13": 10916,
    "X-n139-k10": 13590,
    "X-n143-k7": 15700,
    "X-n148-k46": 43448,
    "X-n153-k22": 21220,
    "X-n157-k13": 16876,
    "X-n162-k11": 14138,
    "X-n167-k10": 20557,
    "X-n172-k51": 45607,
    "X-n176-k26": 47812,
    "X-n181-k23": 25569,
    "X-n186-k15": 24145,
    "X-n190-k8": 16980,
    "X-n195-k51": 44225,
    "X-n200-k36": 58578,
    "X-n206-k16": 30656,
    "X-n209-k16": 30138,
    "X-n219-k73": 117595,
    "X-n228-k23": 25742,
    "X-n233-k16": 29119,
    "X-n237-k14": 27042,
    "X-n247-k50": 37274,
    "X-n251-k28": 38684,
}

data_dict_vrptw = {
    "R101": 1637.7,
    "R102": 1466.6,
    "R103": 1208.7,
    "R104": 971.5,
    "R105": 1355.3,
    "R106": 1234.6,
    "R107": 1064.6,
    "R108": 932.1,
    "R109": 1146.9,
    "R110": 1068.0,
    "R111": 1048.7,
    "R112": 948.6,
    "RC101": 1619.8,
    "RC102": 1457.4,
    "RC103": 1258.0,
    "RC104": 1132.3,
    "RC105": 1513.7,
    "RC106": 1372.7,
    "RC107": 1207.8,
    "RC108": 1114.2,
    "RC201": 1261.8,
    "RC202": 1092.3,
    "RC203": 923.7,
    "RC204": 783.5,
    "RC205": 1154.0,
    "RC206": 1051.1,
    "RC207": 962.9,
    "RC208": 776.1,
}

data_dict_cvrp_large = {
    "X-n502-k39": 69226,
    "X-n513-k21": 24201,
    "X-n524-k153": 154593,
    "X-n536-k96": 94846,
    "X-n548-k50": 86700,
    "X-n561-k42": 42717,
    "X-n573-k30": 50673,
    "X-n586-k159": 190316,
    "X-n599-k92": 108451,
    "X-n613-k62": 59535,
    "X-n627-k43": 62164,
    "X-n641-k35": 63682,
    "X-n655-k131": 106780,
    "X-n670-k130": 146332,
    "X-n685-k75": 68205,
    "X-n701-k44": 81923,
    "X-n716-k35": 43373,
    "X-n733-k159": 136187,
    "X-n749-k98": 77269,
    "X-n766-k71": 114417,
    "X-n783-k48": 72386,
    "X-n801-k40": 73305,
    "X-n819-k171": 158121,
    "X-n837-k142": 193737,
    "X-n856-k95": 88965,
    "X-n876-k59": 99299,
    "X-n895-k37": 53860,
    "X-n916-k207": 329179,
    "X-n936-k151": 132715,
    "X-n957-k87": 85465,
    "X-n979-k58": 118976,
    "X-n1001-k43": 72355,
}


path_1000_20_dict = {0:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/CVRP/or_tools_20s_cvrp1000_uniform.pkl",
1:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRP/or_tools_20s_ovrp1000_uniform.pkl",
2:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPB/or_tools_20s_vrpb1000_uniform.pkl",
3:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPL/or_tools_20s_vrpl1000_uniform.pkl",
4:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPTW/or_tools_20s_vrptw1000_uniform.pkl",
5:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPTW/or_tools_20s_ovrptw1000_uniform.pkl",
6:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPB/or_tools_20s_ovrpb1000_uniform.pkl",
7:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPL/or_tools_20s_ovrpl1000_uniform.pkl",
8:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPBL/or_tools_20s_vrpbl1000_uniform.pkl",
9:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPBTW/or_tools_20s_vrpbtw1000_uniform.pkl",
10: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPLTW/or_tools_20s_vrpltw1000_uniform.pkl",
11: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPBL/or_tools_20s_ovrpbl1000_uniform.pkl",
12: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPBTW/or_tools_20s_ovrpbtw1000_uniform.pkl",
13: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPLTW/or_tools_20s_ovrpltw1000_uniform.pkl",
14: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPBLTW/or_tools_20s_vrpbltw1000_uniform.pkl",
15: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPBLTW/or_tools_20s_ovrpbltw1000_uniform.pkl"
}

path_1000_200_dict = {0:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/CVRP/or_tools_200s_cvrp1000_uniform.pkl",
1:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRP/or_tools_200s_ovrp1000_uniform.pkl",
2:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPB/or_tools_200s_vrpb1000_uniform.pkl",
3:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPL/or_tools_200s_vrpl1000_uniform.pkl",
4:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPTW/or_tools_200s_vrptw1000_uniform.pkl",
5:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPTW/or_tools_200s_ovrptw1000_uniform.pkl",
6:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPB/or_tools_200s_ovrpb1000_uniform.pkl",
7:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPL/or_tools_200s_ovrpl1000_uniform.pkl",
8:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPBL/or_tools_200s_vrpbl1000_uniform.pkl",
9:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPBTW/or_tools_200s_vrpbtw1000_uniform.pkl",
10: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPLTW/or_tools_200s_vrpltw1000_uniform.pkl",
11: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPBL/or_tools_200s_ovrpbl1000_uniform.pkl",
12: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPBTW/or_tools_200s_ovrpbtw1000_uniform.pkl",
13: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPLTW/or_tools_200s_ovrpltw1000_uniform.pkl",
14: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPBLTW/or_tools_200s_vrpbltw1000_uniform.pkl",
15: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPBLTW/or_tools_200s_ovrpbltw1000_uniform.pkl"
}

path_2000_dict = {
0:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/CVRP/or_tools_200s_cvrp2000_uniform.pkl",
1:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRP/or_tools_200s_ovrp2000_uniform.pkl",
2:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPB/or_tools_200s_vrpb2000_uniform.pkl",
3:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPL/or_tools_200s_vrpl2000_uniform.pkl",
4:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPTW/or_tools_200s_vrptw2000_uniform.pkl",
5:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPTW/or_tools_200s_ovrptw2000_uniform.pkl",
6:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPB/or_tools_200s_ovrpb2000_uniform.pkl",
7:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPL/or_tools_200s_ovrpl2000_uniform.pkl",
8:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPBL/or_tools_200s_vrpbl2000_uniform.pkl",
9:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPBTW/or_tools_200s_vrpbtw2000_uniform.pkl",
10: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPLTW/or_tools_200s_vrpltw2000_uniform.pkl",
11: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPBL/or_tools_200s_ovrpbl2000_uniform.pkl",
12: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPBTW/or_tools_200s_ovrpbtw2000_uniform.pkl",
13: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPLTW/or_tools_200s_ovrpltw2000_uniform.pkl",
14: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPBLTW/or_tools_200s_vrpbltw2000_uniform.pkl",
15: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPBLTW/or_tools_200s_ovrpbltw2000_uniform.pkl"
}

path_3000_dict = {
0:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_0.pkl",
1:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_1.pkl",
2:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_2.pkl",
3:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_3.pkl",
4:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_4.pkl",
5:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_5.pkl",
6:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_6.pkl",
7:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_7.pkl",
8:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_8.pkl",
9:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_9.pkl",
10: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_10.pkl",
11: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_11.pkl",
12: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_12.pkl",
13: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_13.pkl",
14: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_14.pkl",
15: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/3000_128_15.pkl"
}

path_4000_dict = {
0:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/CVRP/or_tools_200s_cvrp4000_uniform.pkl",
1:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRP/or_tools_200s_ovrp4000_uniform.pkl",
2:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPB/or_tools_200s_vrpb4000_uniform.pkl",
3:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPL/or_tools_200s_vrpl4000_uniform.pkl",
4:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPTW/or_tools_200s_vrptw4000_uniform.pkl",
5:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPTW/or_tools_200s_ovrptw4000_uniform.pkl",
6:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPB/or_tools_200s_ovrpb4000_uniform.pkl",
7:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPL/or_tools_200s_ovrpl4000_uniform.pkl",
8:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPBL/or_tools_200s_vrpbl4000_uniform.pkl",
9:  "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPBTW/or_tools_200s_vrpbtw4000_uniform.pkl",
10: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPLTW/or_tools_200s_vrpltw4000_uniform.pkl",
11: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPBL/or_tools_200s_ovrpbl4000_uniform.pkl",
12: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPBTW/or_tools_200s_ovrpbtw4000_uniform.pkl",
13: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPLTW/or_tools_200s_ovrpltw4000_uniform.pkl",
14: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/VRPBLTW/or_tools_200s_vrpbltw4000_uniform.pkl",
15: "/home/suyu/suyu/data_distill/test/Routing-MVMoE/data/OVRPBLTW/or_tools_200s_ovrpbltw4000_uniform.pkl"
}


class Tester:
    def __init__(self, args, env_params, model_params, tester_params):

        # save arguments
        self.args = args
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # ENV, MODEL, & Load checkpoint
        self.envs = get_env(self.args.problem)  # Env Class
        self.device = args.device
        self.checkpoint = torch.load(args.checkpoint, map_location=self.device)
        self.model_params['problem'] = self.checkpoint['problem']  # training problem of the checkpoint
        self.model = get_model(self.args.model_type)(**self.model_params)
        self.fine_tune_model = get_model(self.args.model_type)(**self.model_params)
        num_param(self.model)
        self.model.load_state_dict(self.checkpoint['model_state_dict'], strict=True)
        #assert 1 == 0
        self.model.eval_type = "argmax"
        print(">> Checkpoint (Epoch: {}) Loaded!".format(self.checkpoint['epoch']))
        
        # load dataset
        if tester_params['test_set_path'] is None or tester_params['test_set_path'].endswith(".pkl"):
            self.data_dir = "./data"
        else:
            # for solving benchmark instances
            self.path_list = [os.path.join(tester_params['test_set_path'], f) for f in sorted(os.listdir(tester_params['test_set_path']))] \
                if os.path.isdir(tester_params['test_set_path']) else [tester_params['test_set_path']]
            assert self.path_list[-1].endswith(".vrp") or self.path_list[-1].endswith(".txt"), "Unsupported file types."

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        gap = []
        for env_id, env_class in enumerate(self.envs):
            start_time = time.time()
            kkt = 0
            co = 0
            if self.tester_params['test_set_path'] is None or self.tester_params['test_set_path'].endswith(".pkl"):
                compute_gap = not (self.tester_params['test_set_path'] is not None and self.tester_params['test_set_opt_sol_path'] is None)
                if self.tester_params['fine_tune_epochs'] > 0:
                    self._test(self.model, env_class, compute_gap=compute_gap)
                    self._fine_tune(self.model, env_class)
                else:
                    self._test(self.model, env_class, compute_gap=compute_gap, env_id=env_id)
                    
            else:
                for path in self.path_list:
                    if env_class is envs.CVRPEnv:

                            no_aug_score, aug_score = self._solve_cvrplib(self.model, path, env_class)
                            #gap.append((aug_score-data_dict_cvrp[path[62:-4]])/data_dict_cvrp[path[62:-4]])
                            #print((aug_score-data_dict_cvrp[path[62:-4]])/data_dict_cvrp[path[62:-4]])
                            gap.append((aug_score-data_dict_cvrp_large[path[64:-8]])/data_dict_cvrp_large[path[64:-8]])
                            print((aug_score-data_dict_cvrp_large[path[64:-8]])/data_dict_cvrp_large[path[64:-8]])

                    elif env_class is envs.VRPTWEnv:
                        no_aug_score, aug_score = self._solve_cvrptwlib(self.model, path, env_class)
                        gap.append((aug_score-data_dict_vrptw[path[69:-4]])/data_dict_vrptw[path[69:-4]])
                        print((aug_score-data_dict_vrptw[path[69:-4]])/data_dict_vrptw[path[69:-4]])
                    else:
                        raise NotImplementedError
                    
                print("GAP:{}".format(sum(gap)/len(gap)))
            
            print(kkt)

            print(">> Evaluation finished within {:.2f}s\n".format(time.time() - start_time))

    def _test(self, model, env_class, compute_gap=False, env_id=None):

        print(env_class)
        self.time_estimator.reset()
        env = env_class(**self.env_params)
        score_AM, gap_AM = AverageMeter(), AverageMeter()
        aug_score_AM, aug_gap_AM = AverageMeter(), AverageMeter()
        scores, aug_scores = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
        self.tester_params['test_episodes'] = 128 // 8 
        episode, test_num_episode = 0, self.tester_params['test_episodes']
        
        data_path = self.tester_params['test_set_path'] if self.tester_params['test_set_path'] \
            else os.path.join(self.data_dir, env.problem, "{}{}_uniform.pkl".format(env.problem.lower(), env.problem_size))
        
        self.tester_params['test_batch_size'] = 3
        #t = 0
        rec = []
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            
            data = env.load_dataset(data_path, offset=episode, num_samples=batch_size)
            
            score, aug_score, all_score, all_aug_score = self._test_one_batch(model, data, env)
            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            scores = torch.cat((scores, all_score), dim=0)
            aug_scores = torch.cat((aug_scores, all_aug_score), dim=0)
            
            if compute_gap:
                #opt_sol_path = self.tester_params['test_set_opt_sol_path'] if self.tester_params['test_set_opt_sol_path'] \
                #    else get_opt_sol_path(os.path.join(self.data_dir, env.problem), env.problem, env.problem_size)
                opt_sol_path = path_4000_dict[env_id]
                #opt_sol_path_20 = path_1000_20_dict[env_id]
                opt_sol = load_dataset(opt_sol_path, disable_print=True)[episode: episode + batch_size]  # [(obj, route), ...]
                #opt_sol_20 = load_dataset(opt_sol_path_20, disable_print=True)[episode: episode + batch_size]
                
                opt_sol = [i[0] for i in opt_sol if i[0] is not None]
                #opt_sol_20 = [i[0] for i in opt_sol_20]
                #gap = [(opt_sol_20[i].item() - opt_sol[i]) / opt_sol[i] * 100 for i in range(batch_size)]
                #rec.append(sum(opt_sol)/(len(opt_sol)))
            #t += batch_size
            #print(t)
                           #if i[0] is not None]
                batch_size = len(opt_sol)
                #if batch_size == 0:
                #    batch_size = 1
                gap = [(all_score[i].item() - opt_sol[i]) / opt_sol[i] * 100 for i in range(batch_size)]
                aug_gap = [(all_aug_score[i].item() - opt_sol[i]) / opt_sol[i] * 100 for i in range(batch_size)]
                gap_AM.update(sum(gap)/batch_size, batch_size)
                aug_gap_AM.update(sum(aug_gap)/batch_size, batch_size)

            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            print("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                print(" \n*** Test Done on {} *** ".format(env.problem))
                print(" NO-AUG SCORE: {:.4f}, Gap: {:.4f} ".format(score_AM.avg, gap_AM.avg))
                print(" AUGMENTATION SCORE: {:.4f}, Gap: {:.4f} ".format(aug_score_AM.avg, aug_gap_AM.avg))
                print("{:.3f} ({:.3f}%)".format(score_AM.avg, gap_AM.avg))
                print("{:.3f} ({:.3f}%)".format(aug_score_AM.avg, aug_gap_AM.avg))
        
        #print(sum(rec) / len(rec))
        return scores, aug_scores
        #return sum(rec) / len(rec)

    def _test_one_batch(self, model, test_data, env):
        aug_factor = self.tester_params['aug_factor']
        batch_size = test_data.size(0) if isinstance(test_data, torch.Tensor) else test_data[-1].size(0)
        sample_size = self.tester_params['sample_size'] if self.model_params['eval_type'] == "softmax" else 1

        # Sampling: augment data based on sample_size: [batch_size, ...] -> [batch_size x sample_size, ...]
        if self.model_params['eval_type'] == "softmax":
            test_data = list(test_data)
            for i, data in enumerate(test_data):
                if data.dim() == 1:
                    test_data[i] = data.repeat(sample_size)
                elif data.dim() == 2:
                    test_data[i] = data.repeat(sample_size, 1)
                elif data.dim() == 3:
                    test_data[i] = data.repeat(sample_size, 1, 1)

        # Ready
        model.eval()
        with torch.no_grad():
            env.load_problems(batch_size, problems=test_data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

        # POMO Rollout
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)

        # Return
        aug_reward = reward.reshape(aug_factor * sample_size, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
        no_aug_score_mean = no_aug_score.mean()

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value
        aug_score_mean = aug_score.mean()

        return no_aug_score_mean.item(), aug_score_mean.item(), no_aug_score, aug_score

    def _fine_tune(self, model, env_class):
        self.fine_tune_model.load_state_dict(model.state_dict(), strict=True)
        optimizer = Optimizer(self.fine_tune_model.parameters(), lr=self.tester_params['lr'], weight_decay=self.tester_params['weight_decay'])
        # optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        env = env_class(**self.env_params)
        fine_tune_episode = self.tester_params['fine_tune_episodes']

        for i in range(self.tester_params['fine_tune_epochs']):
            episode = 0
            while episode < fine_tune_episode:
                remaining = fine_tune_episode - episode
                batch_size = min(self.tester_params['fine_tune_batch_size'], remaining)
                data = env.get_random_problems(batch_size, self.env_params["problem_size"])
                self._fine_tune_one_batch(self.fine_tune_model, data, env, optimizer)
                episode += batch_size

            print("\n>> Fine-Tuning Epoch {} Finished. Staring Evaluation ...".format(i + 1))
            self._test(self.fine_tune_model, env_class, compute_gap=True)

    def _fine_tune_one_batch(self, model, data, env, optimizer):
        model.train()
        model.set_eval_type(self.model_params["eval_type"])
        aug_factor = self.tester_params['fine_tune_aug_factor']
        batch_size = data.size(0) * aug_factor if isinstance(data, torch.Tensor) else data[-1].size(0) * aug_factor
        env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)
        prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        state, reward, done = env.pre_step()
        while not done:
            selected, prob = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        advantage = reward - reward.float().mean(dim=1, keepdims=True)  # (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        loss_mean = loss.mean()

        if hasattr(model, "aux_loss"):
            loss_mean = loss_mean + model.aux_loss  # add aux(moe)_loss for load balancing (default coefficient: 1e-2)

        # Step & Return
        model.zero_grad()
        loss_mean.backward()
        optimizer.step()

    def _solve_cvrplib(self, model, path, env_class):
        """
            Solving one instance with CVRPLIB format.
        """
        file = open(path, "r")
        lines = [ll.strip() for ll in file]
        i = 0
        
        while i < len(lines):
            line = lines[i]
            if line.startswith("DIMENSION"):
                dimension = int(line.split(':')[1])
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                i = i + dimension
            elif line.startswith('DEMAND_SECTION'):
                demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                i = i + dimension
            i += 1
        original_locations = locations[:, 1:]
        original_locations = np.expand_dims(original_locations, axis=0)  # [1, n+1, 2]
        loc_scaler = 1000
        assert original_locations.max() <= loc_scaler, ">> Scaler is too small"
        locations = original_locations / loc_scaler  # [1, n+1, 2]: Scale location coordinates to [0, 1]
        depot_xy, node_xy = torch.Tensor(locations[:, :1, :]), torch.Tensor(locations[:, 1:, :])
        node_demand = torch.Tensor(demand[1:, 1:].reshape((1, -1))) / capacity  # [1, n]
        
        env_params = {'problem_size': node_xy.size(1), 'pomo_size': node_xy.size(1), 'loc_scaler': loc_scaler, 'device': self.device}
        env = env_class(**env_params)
        data = (depot_xy, node_xy, node_demand)
        _, _, no_aug_score, aug_score = self._test_one_batch(model, data, env)
        no_aug_score = torch.round(no_aug_score * loc_scaler).long()
        aug_score = torch.round(aug_score * loc_scaler).long()
        print(">> Finish solving {} -> no_aug: {} aug: {}".format(path, no_aug_score, aug_score))
        
        return no_aug_score, aug_score

    def _solve_cvrptwlib(self, model, path, env_class):
        """
            Solving one instance with VRPTW benchmark (e.g., Solomon) format.
        """
        file = open(path, "r")
        lines = [ll.strip() for ll in file]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("NUMBER"):
                line = lines[i+1]
                vehicle_number = int(line.split(' ')[0])  # TODO: check vehicle number constraint
                capacity = int(line.split(' ')[-1])
            elif line.startswith("CUST NO."):
                data = np.loadtxt(lines[i + 1:], dtype=int)
                break
            i += 1
        original_locations = data[:, 1:3]
        original_locations = np.expand_dims(original_locations, axis=0)  # [1, n+1, 2]
        scaler = max(original_locations.max(), data[0, 5] / 3.)  # we set the time window of the depot node as [0, 3]
        # scaler = 1000
        assert original_locations.max() <= scaler, ">> Scaler is too small for {}".format(path)
        locations = original_locations / scaler  # [1, n+1, 2]: Scale location coordinates to [0, 1]
        depot_xy, node_xy = torch.Tensor(locations[:, :1, :]), torch.Tensor(locations[:, 1:, :])
        node_demand = torch.Tensor(data[1:, 3].reshape((1, -1))) / capacity  # [1, n]
        service_time = torch.Tensor(data[1:, -1].reshape((1, -1))) / scaler  # [1, n]
        tw_start = torch.Tensor(data[1:, 4].reshape((1, -1))) / scaler  # [1, n]
        tw_end = torch.Tensor(data[1:, 5].reshape((1, -1))) / scaler  # [1, n]

        env_params = {'problem_size': node_xy.size(1), 'pomo_size': node_xy.size(1), 'loc_scaler': scaler, 'device': self.device}
        env = env_class(**env_params)
        env.depot_end = data[0, 5] / scaler
        data = (depot_xy, node_xy, node_demand, service_time, tw_start, tw_end)
        _, _, no_aug_score, aug_score = self._test_one_batch(model, data, env)

        # Check distance
        original_locations = torch.Tensor(original_locations)
        depot_xy = env.augment_xy_data_by_8_fold(original_locations[:, :1, :])
        node_xy = env.augment_xy_data_by_8_fold(original_locations[:, 1:, :])
        original_locations = torch.cat((depot_xy, node_xy), dim=1)
        gathering_index = env.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = original_locations[:, None, :, :].expand(-1, env_params["pomo_size"], -1, -1)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        travel_distances = segment_lengths.sum(2)
        # print(travel_distances.min())

        no_aug_score = torch.round(no_aug_score * scaler).long()
        aug_score = torch.round(aug_score * scaler).long()
        print(">> Finish solving {} -> no_aug: {} aug: {} real: {}".format(path, no_aug_score, aug_score, travel_distances.min()))

        return no_aug_score, aug_score
