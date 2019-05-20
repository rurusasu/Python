%����NN�́A�񎟋Ȗ�z=ax^2+by^2+cxy+dx+ey+f ���ߎ��ł���悤�Ɋw�K����܂��B
%�w�K�̂��߂̌P���f�[�^(x,y,z)��trainingset1�Ŏ��O�ɍ���Ă����܂��B
%sigmoid.m�ł̓V�O���C�h�֐����`���O���t�̌`����m�F����
clear all;%���[�N�̈���N���A
format long;%long�t�H�[�}�b�g�Ő��l��\������
load NN1_data.mat;%���O�ɍ쐬���Ă������P���f�[�^�idata�j�̓ǂݍ���
ACTIVATION = 1 ;%0:Sigmoid function, 1:ReLU function
ReLU_GAIN=0.7;
%�w�K�f�[�^�̕W�����istandardization�j:���ς�0�̕��U��1�ƂȂ�悤�ɓ��̓f�[�^�̃V�t�e�B���O�ƃX�P�[�����O���s���܂��B 
data_1(:,1:1)=(data(:,1:1)-mean(data(:,1:1)))/std(data(:,1:1));%���t�f�[�^�̕W��������(�w�K����UP�ɔ��ɏd�v�I)
data_1(:,2:2)=(data(:,2:2)-mean(data(:,2:2)))/std(data(:,2:2));
data_1(:,3:3)=(data(:,3:3)-mean(data(:,3:3)))/std(data(:,3:3));
%data_1�ɂ͕W�������ꂽ�l���i�[����Ă���B
%�W�������ꂽ�w�K�f�[�^�����ƂɁA���K�������inormalization�j���s��
data_(:,1:1)=data_1(:,1:1)/max(abs(data_1(:,1:1)));%���t�f�[�^�̐��K������(�w�K����UP�ɔ��ɏd�v�I)
data_(:,2:2)=data_1(:,2:2)/max(abs(data_1(:,2:2)));
data_(:,3:3)=data_1(:,3:3)/max(abs(data_1(:,3:3)));
%data_�ɂ͍X�ɐ��K�����ꂽ�l���i�[����Ă���B
%Batch Normalization �Ƃ́H
data_tmp = data_;%�~�j�o�b�`�����Ŏg�p���邽�߂ɁA�W�����Ɛ��K�����s��ꂽdata_��data_tmp�ɕۑ�
if ACTIVATION ==1
    % for ReLU
    eta=0.0001; %�d�݂̊w�K�W��(learning rate for weights),ReLU�̏ꍇ�A�l��傫������ƌ덷���������Ȃ��B
    beta=0.0001; %臒l�̊w�K�W��(learning rate for threshold)(�������֐�����͎���Ŕ����ړ�)
    eta_myu=0.01;%�������iinertia�j�Ƃ͑O��̏d�݂̍X�V�ʂ𔽉f��������x��\��
    beta_myu=0.01;%�������iinertia�j�Ƃ͑O��̏d�݂̍X�V�ʂ𔽉f��������x��\��
else
    % for Sigmoid
    eta=0.1; %�d�݂̊w�K�W��(learning rate for weights),ReLU�̏ꍇ�A�l��傫������ƌ덷���������Ȃ��B
    beta=0.1; %臒l�̊w�K�W��(learning rate for threshold)(�������֐�����͎���Ŕ����ړ�)
    eta_myu=0.1;%�������iinertia�j�Ƃ͑O��̏d�݂̍X�V�ʂ𔽉f��������x��\��
    beta_myu=0.1;%�������iinertia�j�Ƃ͑O��̏d�݂̍X�V�ʂ𔽉f��������x��\��
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%myu=0.3;%������(inertia)�Ƃ͑O��̏d�݂̍X�V�ʂ𔽉f��������x�A�O��̏����g��
N1 = 2 ;%���͂̐��ix,y�j
N2 = 50 ;%��Q�w�i��P�B��w�j�̃j���[����(���j�b�g)�̐�
N3 = 50 ;%��R�w�i��Q�B��w�j�̃j���[�����̐�
N4 = 50 ;%��S�w�i��R�B��w�j�̃j���[�����̐�
N5 = 1 ;%�o�͂̐��iz�j
%
K=2;
w1 = K*(ones(N1,N2)*0.5 - rand(N1,N2)) ;%-1����+1�͈̔͂ŏd�݂̏�����
w2 = K*(ones(N2,N3)*0.5 - rand(N2,N3)) ;%�d�݂̏�����
w3 = K*(ones(N3,N4)*0.5 - rand(N3,N4)) ;%�d�݂̏�����
w4 = K*(ones(N4,N5)*0.5 - rand(N4,N5)) ;%�d�݂̏�����
%
th2 = K*(ones(N2,1)*0.5 - rand(N2,1)) ;%臒l�̏�����
th3 = K*(ones(N3,1)*0.5 - rand(N3,1)) ;%臒l�̏�����
th4 = K*(ones(N4,1)*0.5 - rand(N4,1)) ;%臒l�̏�����
th5 = K*(ones(N5,1)*0.5 - rand(N5,1)) ;%臒l�̏�����
%
w1_1 = w1 ;%�ŏ��̏d�݂̕ۑ�
w2_1 = w2 ;%�ŏ��̏d�݂̕ۑ�
w3_1 = w3 ;%�ŏ��̏d�݂̕ۑ�
w4_1 = w4 ;%�ŏ��̏d�݂̕ۑ�
%
th2_1 = th2 ;%�ŏ���臒l�̕ۑ�
th3_1 = th3 ;%�ŏ���臒l�̕ۑ�
th4_1 = th4 ;%�ŏ���臒l�̕ۑ�
th5_1 = th5 ;%�ŏ���臒l�̕ۑ�(���g�p�H)
%
s1 = zeros(N1,1) ;%��1�w�̃j���[�����̏�ԃx�N�g���̏������i0�ŃN���A�����ۑ�����m�ہj
x1 = zeros(N1,1) ;%��1�w�̃j���[�����̏o�̓x�N�g���̏�����
s2 = zeros(N2,1) ;%��2�w�̃j���[�����̏�ԃx�N�g���̏�����
x2 = zeros(N2,1) ;%��2�w�̃j���[�����̏o�̓x�N�g���̏�����
s3 = zeros(N3,1) ;%��3�w�̃j���[�����̏�ԃx�N�g���̏�����
x3 = zeros(N3,1) ;%��3�w�̃j���[�����̏o�̓x�N�g���̏�����
s4 = zeros(N4,1) ;%��4�w�̃j���[�����̏�ԃx�N�g���̏�����
x4 = zeros(N4,1) ;%��4�w�̃j���[�����̏o�̓x�N�g���̏�����
s5 = zeros(N5,1) ;%��5�w�̃j���[�����̏�ԃx�N�g���̏�����
x5 = zeros(N5,1) ;%��5�w�̃j���[�����̏o�̓x�N�g���̏�����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%�f���^�Ƃ����ʂ������m�F����(BP�@�Ŏg�p�����)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%BP�@�Ŏg�p�����f���^�Ƃ����ʂ������͊m�F����K�v������B
%delta5 = zeros(N5,1) ;%BP�@�ŕK�v�ƂȂ邻�ꂼ��̃j���[�����̃f���^�ƌĂ΂��l�̕ۑ�����m��
%delta4 = zeros(N4,1) ;%BP�@�ŕK�v�ƂȂ邻�ꂼ��̃j���[�����̃f���^�ƌĂ΂��l�̕ۑ�����m��
%delta3 = zeros(N3,1) ;%BP�@�ŕK�v�ƂȂ邻�ꂼ��̃j���[�����̃f���^�ƌĂ΂��l�̕ۑ�����m��
%delta2 = zeros(N2,1) ;%BP�@�ŕK�v�ƂȂ邻�ꂼ��̃j���[�����̃f���^�ƌĂ΂��l�̕ۑ�����m��
%delta1 = zeros(N1,1) ;%BP�@�ŕK�v�ƂȂ邻�ꂼ��̃j���[�����̃f���^�ƌĂ΂��l�̕ۑ�����m��
Batch_size = size(data(:,1:1));%�P���f�[�^���̃T���v������Batch_size�Ɋi�[
Iteration_limit = 1000000;%Minibatch_size�̃f�[�^���g�����ő�w�K��
%Minibatch_size = Batch_size(1,1);
Minibatch_size = 10000;%Batch_size���璊�o�����T�C�Y
Ev_buff = zeros(Iteration_limit,1);%�w�K�̐i���󋵁i�P���f�[�^���̂P�T���v��������̌덷�j��ۑ�����o�b�t�@�ł���A�덷�͖ڕW�o�͂�NN����̏o�͂Ƃ̍��ł���B
for iteration = 1:Iteration_limit %�S�Ă̌P���f�[�^���g�����O�����v�Z��BP�@�ɂ��������v�Z���AMax_epoch�̉񐔂����s���B
    data_=data_tmp;%�W�����Ɛ��K�������ꂽ�P���f�[�^���Z�b�g
    data_ = datasample(data_(:,:),Minibatch_size);%�S�P���f�[�^����Minibatch_size�������o
    [Row Column] = size(data_);%���o���ꂽ�P���f�[�^�̍s���Ɨ񐔂��擾
    %�W�����Ɛ��K�����ꂽ�f�[�^�ۑ��p
    x5_buff = zeros(Row,N5) ;%NN����̏o�́i��5�w�̃��j�b�g����̏o�́j��ۑ����Ă����o�b�t�@���N���A
    x5_err_buff = zeros(Row,N5) ;%NN����̏o�́i��5�w�̃��j�b�g����̏o�́j�ƌP���f�[�^�Ƃ̌덷��ۑ����Ă����o�b�t�@���N���A
    %�W�����Ɛ��K�������O�̃I���W�i���f�[�^�ۑ��p
    x5_buff2 = zeros(Row,N5) ;%NN����̏o�́i��5�w�̃��j�b�g����̏o�́j��ۑ����Ă����o�b�t�@���N���A
    x5_err_buff2 = zeros(Row,N5) ;%NN����̏o�́i��5�w�̃��j�b�g����̏o�́j�ƌP���f�[�^�Ƃ̌덷��ۑ����Ă����o�b�t�@���N���A
    for sample_counter = 1:Row %Row�̉񐔁iMinibatch_size�Ŏw�肳�ꂽ�P���f�[�^�j�����O�����v�Z��BP�@�ɂ��������v�Z�i�d�݂̍X�V�j���s���B
        %���w
        s1(1)=data_(sample_counter:sample_counter,1);%sample_counter�͌P���f�[�^�̑扽�s����\���B
        s1(2)=data_(sample_counter:sample_counter,2);
        x1(1)=s1(1);%���͑w�ł͂��̂܂܏o�͂����
        x1(2)=s1(2);
        %���w     
        for i=1:N2%i�͑�2�w�̃��j�b�g��
            s2(i) = 0 ;%��ԗʂ����߂邽�߂ɂ܂��͏���������K�v������B
            for j = 1:N1%j�͑�1�w�̃��j�b�g��
                s2(i) = s2(i) + w1(j,i)*x1(j) ;%��2�w�̏�ԃx�N�g���̐��������߂�B
            end
            s2(i) = s2(i) + th2(i) ;%��ԗʂ�臒l��������B->�@�������֐��ւ̓��͂ƂȂ�B
            %
            if ACTIVATION == 0
                x2(i) = 1/(1+exp(-s2(i)))-1/2 ;%�������֐��i�V�O���C�h�֐��j����̏o�͂����߂�B
            else
                if s2(i) > 0                    %�������֐��iReLU�j����̏o�͂����߂�B
                    x2(i) = s2(i);
                else
                    x2(i) = ReLU_GAIN*s2(i);
                end
            end
        end
        %x2= x2/max(abs(x2));
        %��O�w
        for i=1:N3%i�͑�3�w�̃��j�b�g��
            s3(i) = 0 ;%��ԗʂ����߂邽�߂ɂ܂��͏���������K�v������B
            for j = 1:N2%j�͑�2�w�̃��j�b�g��
                s3(i) = s3(i) + w2(j,i)*x2(j) ;%��3�w�̏�ԃx�N�g���̐��������߂�B
            end
            s3(i) = s3(i) + th3(i) ;%��ԗʂ�臒l��������B->�@�������֐��ւ̓��͂ƂȂ�B
            %
            if ACTIVATION == 0
                x3(i) = 1/(1+exp(-s3(i))) -1/2;%�������֐�����̏o�͂����߂�B
            else
                if s3(i) > 0
                    x3(i) = s3(i);
                else
                    x3(i) = ReLU_GAIN*s3(i);
                end
            end
        end  
        %x3= x3/max(abs(x3));
        %��l�w     
        for i=1:N4%i�͑�4�w�̃��j�b�g��
            s4(i) = 0 ;%��ԗʂ����߂邽�߂ɂ܂��͏���������K�v������B
            for j = 1:N3%j�͑�3�w�̃��j�b�g��
                s4(i) = s4(i) + w3(j,i)*x3(j) ;%��4�w�̏�ԃx�N�g���̐��������߂�B
            end
            s4(i) = s4(i) + th4(i) ;%��ԗʂ�臒l��������B
            %
            if ACTIVATION == 0
                x4(i) = 1/(1+exp(-s4(i))) -1/2;%�������֐�����̏o�͂����߂�B
             else
                 if s4(i) > 0
                     x4(i) = s4(i);
                 else
                     x4(i) = ReLU_GAIN*s4(i);
                 end
             end
        end
        %��ܑw     
        for i=1:N5%i�͑�5�w�̃��j�b�g��
            s5(i) = 0 ;%��ԗʂ����߂邽�߂ɂ܂��͏���������K�v������B
            for j = 1:N4%j�͑�4�w�̃��j�b�g��
                s5(i) = s5(i) + w4(j,i)*x4(j) ;%��5�w�̏�ԃx�N�g���̐��������߂�B
            end
            x5(i)=s5(i);
            %x5(i) = 1/(1+exp(-s5(i))) -1/2;%�������֐�����̏o�͂����߂�B            
        end
        x5_d=data_(sample_counter,3);%BP�@�ł́Ax5��x5_d(���O�ɗp�ӂ��A�W�����A���K�����ꂽ�P���f�[�^)�ɋ߂Â��悤�ɏd�݂����������B
        for i=1:N5
            %�W�����Ɛ��K���������ꂽ�f�[�^�ɂ��o�͂ƌ덷
            x5_buff(sample_counter,i)=x5(i);%NN����̏o�͂�ۑ����Ă��邾���B�Ȃ��Ă�NN�̊w�K�͂ł��܂��B
            %x5_err_buff(sample_counter,i)=abs(x5_d-x5(i));%NN����̏o�͂ƌP���f�[�^�Ƃ̌덷��ۑ����Ă���
            x5_err_buff(sample_counter,i)=0.5*(x5(i)-x5_d)^2;
            %�I���W�i���̃f�[�^�ɂ��o�͂ƌ덷
            x5_buff2(sample_counter,i)=x5(i)*max(abs(data_1(:,3:3)))*std(data(:,3:3))+mean(data(:,3:3));
            x5_err_buff2(sample_counter,i)=abs(data(sample_counter,3)-x5_buff2(sample_counter,i));
        end
        %�֐�weight�ł́ABP�@�ɂ��d�݂��X�V����܂��Bx5��x5_d�Ō덷������܂��BBP�@�ł͈�ʉ��f���^���[�����K�p����܂��B
        %��ʓI�ɁA[�߂�l]�@���@�֐��i�����j�G�̌`���ƂȂ�B
        [w1,w2,w3,w4,th2,th3,th4,th5,w1_1,w2_1,w3_1,w4_1,th2_1,th3_1,th4_1,th5_1] = weight_20180329_5layers(w1,w2,w3,w4,th2,th3,th4,th5,eta,beta,x1,x2,x3,x4,x5,x5_d,s2,s3,s4,s5,w1_1,w2_1,w3_1,w4_1,eta_myu,beta_myu,N1,N2,N3,N4,N5,th2_1,th3_1,th4_1,th5_1,ACTIVATION,ReLU_GAIN) ;
    end
    err_sum =0;
    for i=1:sample_counter
        for j=1:N5
            err_sum =err_sum +x5_err_buff(i,j);%Minibatch_size�Ŏw�肳�ꂽ�f�[�^���̌덷�̑��a
        end
    end
    Ev_buff(iteration) = err_sum/sample_counter;%�P���f�[�^���̂P�T���v��������̕��ό덷��ۑ�
    if rem(iteration,1)==0%�����ŕ\���p�x��ݒ肵�Ă���B
        Epoch = round(Minibatch_size*iteration/Batch_size(1,1));
        fprintf('\n Epock = %d, Iteration = %g', Epoch,iteration);%���݂̃~�j�o�b�`���s�񐔂�\�� 
        fprintf('\n err_sum/sample_counter = %g', err_sum/sample_counter); 
        plot(Ev_buff(1:iteration));%�w�K�̐i���󋵂�\������
        pause(0.1);%�O���t�\�����m���ɍs�����߂�0.1�b�Ԃ̃|�[�Y
        save NN1_weight w1 w2 w3 w4 th2 th3 th4 th5 Ev_buff N1 N2 N3 N4 N5 eta beta eta_myu beta_myu iteration Minibatch_size ACTIVATION ;%�w�K�����d�݂ƃp�����[�^�Ȃǂ�ۑ�����B
    end
    if err_sum/sample_counter < 0.0001 %���ό덷���������Ȃ�����w�K�̏I���ł��B�@
        save NN1_weight w1 w2 w3 w4 th2 th3 th4 th5 Ev_buff N1 N2 N3 N4 N5 eta beta eta_myu beta_myu iteration Minibatch_size ACTIVATION ;%�w�K�����d�݂ƃp�����[�^�Ȃǂ�ۑ�����B
        break;
    end
end


