import json
import random
import re
import time

from z3 import *

from pearl.policy_learners.sequential_decision_making.soft_actor_critic import SoftActorCritic
# from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import BootstrapReplayBuffer
from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule
from pearl.history_summarization_modules.lstm_history_summarization_module import LSTMHistorySummarizationModule
from pearl.pearl_agent import PearlAgent

import torch
from env_gai_3 import ConstraintSimplificationEnv_test

from test_code_bert_4 import CodeEmbedder, CodeEmbedder_normalize
from test_rl.bert_predictor_2_mask import EnhancedEightClassModel
from test_rl.bert_predictor_mask import SimpleClassifier
from test_rl.test_script.utils import parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict, \
    solve_and_measure_time, model_to_dict, load_dictionary, extract_variables_from_smt2_content, normalize_variables
from test_rl.test_script.online_learning_break import online_learning

start = time.time()


def test_group():
    file_path_list = ['/files/chmod54591','/files/who81686']

    for file_path in file_path_list:
        with open(file_path, 'r') as file:
            # 读取文件所有内容到一个字符串
            smtlib_str = file.read()
        # 解析字符串
        try:
            # 将JSON字符串转换为字典
            dict_obj = json.loads(smtlib_str)
            # print("转换后的字典：", dict_obj)
        except json.JSONDecodeError as e:
            print("解析错误：", e)
        #
        if 'smt-comp' in file_path:
            smtlib_str = dict_obj['smt_script']
        else:
            smtlib_str = dict_obj['script']
        # variables = set()
        variables = extract_variables_from_smt2_content(smtlib_str)
        smtlib_str = normalize_variables(smtlib_str, variables)
        if len(variables) > 40:
            continue
        print("变量列表：")
        for v in variables:
            print(v)

        assertions = parse_smt2_string(smtlib_str)
        solver = Solver()
        for a in assertions:
            solver.add(a)
        timeout = 999999999
        # timeout = 1000
        result, model, time_taken = solve_and_measure_time(solver, timeout)
        print(result, time_taken)
        result_list = [result, time_taken, timeout]
        # if result == sat:
        #     result = 'sat'
        # elif result == unknown:
        #     result = 'unknown'
        # else:
        #     result = 'unsat'

        # result_dict[filepath] = result_list
        if model:
            result_list.append(model_to_dict(model))
        # file_path = key
        # file_path = '/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/ginstall/ginstall307943'

        start_time = time.time()

        embedder = CodeEmbedder_normalize()
        set_seed(0)
        # device = torch.device("cpu")
        # 更改了预测器
        model = SimpleClassifier()
        model_path = 'bert_predictor_mask_best.pth'  # 或者 'bert_predictor_mask_final.pth'
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.eval()
        model_time = EnhancedEightClassModel()
        model_time.load_state_dict(torch.load('bert_predictor_2_mask_best_model.pth'))
        model_time.eval()
        env = ConstraintSimplificationEnv_test(embedder, assertions, model, model_time,smtlib_str, file_path)
        observation, action_space = env.reset()
        action_representation_module = IdentityActionRepresentationModule(
            max_number_actions=action_space.n,
            representation_dim=action_space.action_dim,
        )
        # action_representation_module = OneHotActionTensorRepresentationModule(
        #     max_number_actions=len(env.variables)*20000,
        # )
        # action_representation_module = IdentityActionRepresentationModule(
        #     max_number_actions=len(variables)*20000,
        #     representation_dim=action_space.action_dim,
        # )
        # experiment code
        number_of_steps = 500
        number_of_episodes = 1
        record_period = 1
        # 创建强化学习代理
        print(len(env.variables))
        agent = PearlAgent(
            policy_learner=SoftActorCritic(
                state_dim=768,
                action_space=action_space,
                actor_hidden_dims=[768, 512, 128],
                critic_hidden_dims=[768, 512, 128],
                action_representation_module=action_representation_module,
            ),
            # history_summarization_module=StackingHistorySummarizationModule(
            #     observation_dim=768,
            #     action_dim=len(env.variables) + 1,
            #     history_length=len(env.variables),
            # ),
            history_summarization_module=LSTMHistorySummarizationModule(
                observation_dim=768,
                action_dim=len(env.variables) + 1,
                hidden_dim=768,
                history_length=len(env.variables),  # 和完整结点数相同
            ),
            replay_buffer=FIFOOffPolicyReplayBuffer(10),
            device_id=-1,
        )
        # 训练代理
        info = online_learning(
            agent=agent,
            env=env,
            number_of_episodes=number_of_episodes,
            print_every_x_episodes=1,
            record_period=record_period,
            # learn_after_episode=True,
        )
        del agent
        del env
        torch.cuda.empty_cache()
        # torch.save(info["return"], "BootstrappedDQN-LSTM-return.pt")
        # plt.plot(record_period * np.arange(len(info["return"])), info["return"], label="BootstrappedDQN-LSTM")
        # plt.legend()
        # plt.show()


if __name__ == '__main__':
    test_group()
