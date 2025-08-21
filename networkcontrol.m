%% 参数设置
N = 10; T = 50;
lambda1 = 0.1; lambda2 = 0.1; alpha = 3;

A = randn(N,N)*0.1;     % 已知邻接矩阵
X_obs = randn(N,T);
x0 = X_obs(:,1);

% ------------------------------
% 把 (X,U,B) 拼成一个向量 z
% ------------------------------
nx = N*T; nu = N*T; nb = N; 
z0 = randn(nx+nu+nb,1);   % 初始猜测

% 辅助索引
idxX = 1:nx;
idxU = nx+(1:nu);
idxB = nx+nu+(1:nb);

% 目标函数
fun = @(z) objective_fun(z, A, X_obs, N, T, lambda1, lambda2, idxX, idxU, idxB);

% 约束
nonlcon = @(z) dynamics_constraints(z, A, x0, N, T, alpha, idxX, idxU, idxB);

% 边界 (B relaxed 到 [0,1])
lb = -inf*ones(nx+nu+nb,1);
ub =  inf*ones(nx+nu+nb,1);
lb(idxB) = 0; ub(idxB) = 1;

% 优化
opts = optimoptions('fmincon','Display','iter','Algorithm','sqp');
[z_opt,fval] = fmincon(fun,z0,[],[],[],[],lb,ub,nonlcon,opts);

B_opt = z_opt(idxB);
U_opt = reshape(z_opt(idxU),[N,T]);
X_opt = reshape(z_opt(idxX),[N,T]);

%% --- 函数定义 ---
function f = objective_fun(z,A,X_obs,N,T,lambda1,lambda2,idxX,idxU,idxB)
    X = reshape(z(idxX),[N,T]);
    U = reshape(z(idxU),[N,T]);
    % B 只在约束里出现，这里不用
    f = sum(sum((X - X_obs).^2)) ...
        + lambda1*sum(sum(U.^2)) ...
        + lambda2*sum(sum((U(:,2:T)-U(:,1:T-1)).^2));
end

function [c,ceq] = dynamics_constraints(z,A,x0,N,T,alpha,idxX,idxU,idxB)
    X = reshape(z(idxX),[N,T]);
    U = reshape(z(idxU),[N,T]);
    B = diag(z(idxB));   % 只取对角线
    % 动力学约束
    ceq_dyn = [];
    for t=1:T-1
        ceq_dyn = [ceq_dyn; X(:,t+1) - (A*X(:,t) + B*U(:,t))];
    end
    % 初始条件
    ceq_init = X(:,1) - x0;
    % 节点数约束
    ceq_alpha = sum(z(idxB)) - alpha;
    ceq = [ceq_dyn; ceq_init; ceq_alpha];
    c = [];
end



%%
%% 参数设置
clear all
N = 10; T = 50; alpha = 3;
lambda1 = 0.1; lambda2 = 0.1;
A = randn(N,N)*0.1;
X_obs = randn(N,T);
x0 = X_obs(:,1);

% 初始化
B = eye(N); B = diag(rand(N,1)>0.5);   % 随机选节点
U = randn(N,T);
X = zeros(N,T); X(:,1)=x0;

max_iter = 10;

for iter=1:max_iter
    fprintf('迭代 %d...\n',iter);

    %% Step 1: 固定 B，优化 X,U (凸问题)
    cvx_begin quiet
        variables U(N,T) X(N,T)
        reconstruction_loss = sum_square(vec(X - X_obs));
        input_energy = lambda1*sum_square(vec(U));
        smoothness = 0;
        for t=1:T-1
            smoothness = smoothness + lambda2*sum_square(U(:,t+1)-U(:,t));
        end
        minimize( reconstruction_loss + input_energy + smoothness )
        subject to
            X(:,1) == x0;
            for t=1:T-1
                X(:,t+1) == A*X(:,t) + B*U(:,t);
            end
    cvx_end

    %% Step 2: 固定 U，优化 B,X
    cvx_begin quiet
        variables Bdiag(N) X(N,T)
        B = diag(Bdiag);
        reconstruction_loss = sum_square(vec(X - X_obs));
        minimize( reconstruction_loss )
        subject to
            X(:,1) == x0;
            for t=1:T-1
                X(:,t+1) == A*X(:,t) + B*U(:,t);
            end
            sum(Bdiag) == alpha;
            0 <= Bdiag <= 1;
    cvx_end
end

disp('最终关键节点选择:')
disp(round(diag(B)))