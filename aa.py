# 导入需要使用的库,其中numpy是支持数组和矩阵运算的科学计算库,而matplotlib是绘图库
import numpy as np
import matplotlib.pyplot as plt


class Solver:
    """ 多臂老虎机算法基本框架 """
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.count)  # 每根拉杆的尝试次数
        # self.counts: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        self.regret = 0.  # 当前步的累积懊悔
        self.actions = []  # 维护一个列表,记录每一步的动作
        self.regrets = []  # 维护一个列表,记录每一步的累积懊悔

    def update_regret(self, index):
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        self.regret += self.bandit.best_prob - self.bandit.probs[index]
        self.regrets.append(self.regret)

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        raise NotImplementedError

    def run(self, num_steps):
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps):
            index = self.run_one_step()
            self.counts[index] += 1
            self.actions.append(index)
            self.update_regret(index)



class EpsilonGreedy(Solver):
    """ epsilon贪婪算法,继承Solver类 """
    def __init__(self, bandit, epsilon=0.1, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.count)
        #print(f"self.estimates: {self.estimates}")
        # self.estimates: [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            index = np.random.randint(0, self.bandit.count)  # 随机选择一根拉杆
        else:
            index = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(index)  # 得到本次动作的奖励
        self.estimates[index] += 1. / (self.counts[index] + 1) * (r - self.estimates[index])

        return index




class BernoulliBandit:
    """ 伯努利多臂老虎机,输入K表示拉杆个数 """
    def __init__(self, K):
        self.probs = np.random.uniform(size=K)  # 随机生成K个0～1的数,作为拉动每根拉杆的获奖
        # 概率
        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
        self.count = K
        """
        self.probs: [4.17022005e-01 7.20324493e-01 1.14374817e-04 3.02332573e-01
        1.46755891e-01 9.23385948e-02 1.86260211e-01 3.45560727e-01
        3.96767474e-01 5.38816734e-01]
        best_idx: 1
        self.best_prob: 0.7203244934421581  
        """
       ## print(f"self.probs: {self.probs}\nbest_idx: {self.best_idx}\nself.best_prob: {self.best_prob}\n")

    def step(self, index):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未
        # 获奖）
        if np.random.rand() < self.probs[index]:
            return 1
        else:
            return 0
            
            
            
def plot_results(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称"""
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.count)
    plt.legend()
    plt.show()


np.random.seed(3)
K = 10
bandit_10_arm = BernoulliBandit(K)

print(f"随机生成了一个 {K} 臂伯努利老虎机")
print(f"获奖概率最大的拉杆为{bandit_10_arm.best_idx}号,其获奖概率为 {bandit_10_arm.best_prob}")

epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print(f"epsilon-贪婪算法的累积懊悔为：{epsilon_greedy_solver.regret}")
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])
           
            




