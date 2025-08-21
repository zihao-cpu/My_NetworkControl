%% ��������
N = 10; T = 50;
lambda1 = 0.1; lambda2 = 0.1; alpha = 3;

A = randn(N,N)*0.1;     % ��֪�ڽӾ���
X_obs = randn(N,T);
x0 = X_obs(:,1);

% ------------------------------
% �� (X,U,B) ƴ��һ������ z
% ------------------------------
nx = N*T; nu = N*T; nb = N; 
z0 = randn(nx+nu+nb,1);   % ��ʼ�²�

% ��������
idxX = 1:nx;
idxU = nx+(1:nu);
idxB = nx+nu+(1:nb);

% Ŀ�꺯��
fun = @(z) objective_fun(z, A, X_obs, N, T, lambda1, lambda2, idxX, idxU, idxB);

% Լ��
nonlcon = @(z) dynamics_constraints(z, A, x0, N, T, alpha, idxX, idxU, idxB);

% �߽� (B relaxed �� [0,1])
lb = -inf*ones(nx+nu+nb,1);
ub =  inf*ones(nx+nu+nb,1);
lb(idxB) = 0; ub(idxB) = 1;

% �Ż�
opts = optimoptions('fmincon','Display','iter','Algorithm','sqp');
[z_opt,fval] = fmincon(fun,z0,[],[],[],[],lb,ub,nonlcon,opts);

B_opt = z_opt(idxB);
U_opt = reshape(z_opt(idxU),[N,T]);
X_opt = reshape(z_opt(idxX),[N,T]);

%% --- �������� ---
function f = objective_fun(z,A,X_obs,N,T,lambda1,lambda2,idxX,idxU,idxB)
    X = reshape(z(idxX),[N,T]);
    U = reshape(z(idxU),[N,T]);
    % B ֻ��Լ������֣����ﲻ��
    f = sum(sum((X - X_obs).^2)) ...
        + lambda1*sum(sum(U.^2)) ...
        + lambda2*sum(sum((U(:,2:T)-U(:,1:T-1)).^2));
end

function [c,ceq] = dynamics_constraints(z,A,x0,N,T,alpha,idxX,idxU,idxB)
    X = reshape(z(idxX),[N,T]);
    U = reshape(z(idxU),[N,T]);
    B = diag(z(idxB));   % ֻȡ�Խ���
    % ����ѧԼ��
    ceq_dyn = [];
    for t=1:T-1
        ceq_dyn = [ceq_dyn; X(:,t+1) - (A*X(:,t) + B*U(:,t))];
    end
    % ��ʼ����
    ceq_init = X(:,1) - x0;
    % �ڵ���Լ��
    ceq_alpha = sum(z(idxB)) - alpha;
    ceq = [ceq_dyn; ceq_init; ceq_alpha];
    c = [];
end



%%
%% ��������
clear all
N = 10; T = 50; alpha = 3;
lambda1 = 0.1; lambda2 = 0.1;
A = randn(N,N)*0.1;
X_obs = randn(N,T);
x0 = X_obs(:,1);

% ��ʼ��
B = eye(N); B = diag(rand(N,1)>0.5);   % ���ѡ�ڵ�
U = randn(N,T);
X = zeros(N,T); X(:,1)=x0;

max_iter = 10;

for iter=1:max_iter
    fprintf('���� %d...\n',iter);

    %% Step 1: �̶� B���Ż� X,U (͹����)
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

    %% Step 2: �̶� U���Ż� B,X
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

disp('���չؼ��ڵ�ѡ��:')
disp(round(diag(B)))