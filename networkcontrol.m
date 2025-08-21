%% 参数设置
% "Reverse engineering the brain input: Network control theory to identify cognitive task-related control nodes". Zhichao Liang et al
N = 10;                % 节点数
T = 50;                % 时间长度
alpha = 3;             % 关键节点个数
lambda1 = 0.1;         % 超参数 λ1
lambda2 = 0.1;         % 超参数 λ2
epsilon = 1e-3;        % 松弛参数

% 邻接矩阵 A（实际应用中应从数据获得）
A = randn(N,N)*0.1;

% 观测的脑网络状态 x_hat(t)，假设已知
X_obs = randn(N,T);

% 初始状态
x0 = X_obs(:,1);

%% CVX 优化
cvx_begin quiet
    variables B(N,N) U(N,T) X(N,T)

    % -------- 目标函数 --------
    reconstruction_loss = sum_square(vec(X - X_obs));    % 数据重建误差
    input_energy        = lambda1 * sum_square(vec(U));  % 输入能量约束
    smoothness          = 0;                             % 平滑项
    for t = 1:T-1
        smoothness = smoothness + lambda2 * sum_square(U(:,t+1) - U(:,t));
    end
    
    minimize( reconstruction_loss + input_energy + smoothness )

    % -------- 约束条件 --------
    subject to
        % 初始条件
        X(:,1) == x0;

        % 动力学约束
        for t = 1:T-1
            X(:,t+1) == A*X(:,t) + B*U(:,t);
        end

        % 关键节点数量约束
        sum(diag(B)) == alpha;

        % epsilon-松弛布尔约束
        for i = 1:N
            abs(B(i,i) * (1 - B(i,i))) <= epsilon;
        end
cvx_end

%% 输出结果
disp('优化完成！')
disp('关键节点选择结果 (近似0-1)：')
disp(round(diag(B)))   % 四舍五入得到近似布尔解
