%�o�b�N�v���p�Q�[�V�����A���S���Y���ɂ��d�݂��P������֐��ł���B
function [w1,w2,w3,w4,th2,th3,th4,th5,w1_1,w2_1,w3_1,w4_1,th2_1,th3_1,th4_1,th5_1] = weight(w1,w2,w3,w4,th2,th3,th4,th5,eta,beta,x1,x2,x3,x4,x5,x5_d,s2,s3,s4,s5,w1_1,w2_1,w3_1,w4_1,eta_myu,beta_myu,N1,N2,N3,N4,N5,th2_1,th3_1,th4_1,th5_1,ACTIVATION,ReLU_GAIN) ;

    w1_tmp = w1_1 ;%�O��̏d�݂�ۑ����Ă���(���͑w�Ƒ�Q�w�i��P�B��w�j�Ԃ̏d��)
	w2_tmp = w2_1 ;%��Q�w�i��P�B��w�j�Ƒ�R�w�i��Q�B��w�j�Ԃ̏d��
	w3_tmp = w3_1 ;%��R�w�i��Q�B��w�j�Ƒ�S�w�i��R�B��w�j�Ԃ̏d��
	w4_tmp = w4_1 ;%��S�w�i��R�B��w�j�Ƒ�T�w�i�o�͑w�j�Ԃ̏d��
    th2_tmp = th2_1;%�O���臒l��ۑ����Ă���
    th3_tmp = th3_1;
    th4_tmp = th4_1;
    th5_tmp = th5_1;
    w1_1 = w1 ;%�������̌v�Z�̂��߂ɁA���݂̏d�݂�O��̏d�݂Ƃ��ĕۑ� (���͑w�Ƒ�Q�w�i��P�B��w�j�Ԃ̏d��)����
	w2_1 = w2 ;
	w3_1 = w3 ;
	w4_1 = w4 ;
    th2_1 = th2;%�������̌v�Z�̂��߂ɁA���݂�臒l��O���臒l�Ƃ��ĕۑ�����
    th3_1 = th3;
    th4_1 = th4;
    th5_1 = th5;
    delta5 = zeros(N5,1) ;%BP�@�ŕK�v�ƂȂ邻�ꂼ��̃j���[�����̃f���^�ƌĂ΂��l�̕ۑ�����m��
    delta4 = zeros(N4,1) ;%BP�@�ŕK�v�ƂȂ邻�ꂼ��̃j���[�����̃f���^�ƌĂ΂��l�̕ۑ�����m��
    delta3 = zeros(N3,1) ;%BP�@�ŕK�v�ƂȂ邻�ꂼ��̃j���[�����̃f���^�ƌĂ΂��l�̕ۑ�����m��
    delta2 = zeros(N2,1) ;%BP�@�ŕK�v�ƂȂ邻�ꂼ��̃j���[�����̃f���^�ƌĂ΂��l�̕ۑ�����m��
    %delta1 = zeros(N1,1) ;%BP�@�ŕK�v�ƂȂ邻�ꂼ��̃j���[�����̃f���^�ƌĂ΂��l�̕ۑ�����m��    
%%%
%%%	�X�e�b�v1�F�o�͑w�Ɍ����������W��(w4)���C������B
%%%
   for i=1:N5
        if ACTIVATION == 0
            %�V�O���C�h�֐��̏ꍇ s5(��ܑw�̏o��(Affine + liner))
            delta5(i) = (1/(1+exp(-s5(i))))*(1-1/(1+exp(-s5(i))))*(x5(i) - x5_d(i)) ;%�o�͑w�ɂ�����f���^
        else
            %ReLU�̏ꍇ
            if s5(i) > 0
                delta5(i)=(x5(i) - x5_d(i));
            else
                delta5(i) = ReLU_GAIN*(x5(i) - x5_d(i));
            end
        end
        for j = 1:N4
            w4(j,i) = w4(j,i) - eta * delta5(i)*x4(j) + eta_myu * (w4(j,i) - w4_tmp(j,i)) ;%����������̏d�ݍX�V
        end
        th5(i) = th5(i) - beta * delta5(i)*x4(j) + beta_myu * (th5(i) - th5_tmp(i));%�����������臒l�̍X�V
   end
%%%�@���̑��̌����W���͓��͑w�Ɍ������ď����C�����s���B�덷�̌v�Z�ɒ��ӁI
%%%�@�X�e�b�v�Q�F�����W��(w3)���C������
  for i=1:N4
	sigma = 0 ;%���ꂪ�덷�ƂȂ�B
	for k=1:N5
		sigma = sigma + delta5(k)*w4(i,k) ;%�������덷�̌v�Z�ƂȂ��Ă���B
    end
    if ACTIVATION == 0
        %�V�O���C�h�֐��̏ꍇ
        delta4(i) = (1/(1+exp(-s4(i))))*(1-1/(1+exp(-s4(i))))*sigma ;%�o�͑w�ȊO�̃f���^�̌v�Z
    else     
        %ReLU�̏ꍇ
        if s4(i) > 0
            delta4(i)=sigma;
        else
            delta4(i) = ReLU_GAIN*sigma;
        end
    end
	for j = 1:N3
		w3(j,i) = w3(j,i) - eta * delta4(i)*x3(j) + eta_myu * (w3(j,i) - w3_tmp(j,i)) ;
	end
    th4(i) = th4(i) - beta * delta4(i)*x3(j)+ beta_myu * (th4(i) - th4_tmp(i));%���j�b�g��臒l�̍X�V
  end   
%%%�@���̑��̌����W���͓��͑w�Ɍ������ď����C�����s���B�덷�̌v�Z�ɒ��ӁI
%%%�@�X�e�b�v�R�F�����W��(w2)���C������
  for i=1:N3
	sigma = 0 ;%���ꂪ�덷�ƂȂ�B
	for k=1:N4
		sigma = sigma + delta4(k)*w3(i,k) ;%�������덷�̌v�Z�ƂȂ��Ă���B
    end
    if ACTIVATION == 0
        %�V�O���C�h�֐��̏ꍇ
        delta3(i) = (1/(1+exp(-s3(i))))*(1-1/(1+exp(-s3(i))))*sigma ;%�o�͑w�ȊO�̃f���^�̌v�Z
    else
        %ReLU�̏ꍇ
        if s3(i) > 0
            delta3(i)=sigma;
        else
            delta3(i) = ReLU_GAIN*sigma;
        end
    end
	for j = 1:N2
		w2(j,i) = w2(j,i) - eta * delta3(i)*x2(j) + eta_myu * (w2(j,i) - w2_tmp(j,i)) ;
	end
    th3(i) = th3(i) - beta * delta3(i)*x2(j)+ beta_myu * (th3(i) - th3_tmp(i));%���j�b�g��臒l�̍X�V
  end
  %%%�@�X�e�b�v�S�F�����W��(w1)���C������
  for i=1:N2
	sigma = 0 ;%���ꂪ�덷�ƂȂ�B�܂��͏���������B
	for k=1:N3
		sigma = sigma + delta3(k)*w2(i,k) ;%�o�͑w�ȊO�̌덷�ʂ̌v�Z
    end
    if ACTIVATION == 0
        %�V�O���C�h�֐��̏ꍇ
        delta2(i) = (1/(1+exp(-s2(i))))*(1-1/(1+exp(-s2(i))))*sigma ;
    else
        %ReLU�̏ꍇ
        if s2(i) > 0
            delta2(i)=sigma;
        else
            delta2(i) = ReLU_GAIN*sigma;
        end
    end
	for j = 1:N1
		w1(j,i) = w1(j,i) - eta * delta2(i)*x1(j) + eta_myu * (w1(j,i) - w1_tmp(j,i)) ;
	end
    th2(i) = th2(i) - beta * delta2(i)*x1(j)+ beta_myu * (th2(i) - th2_tmp(i));%臒l�̍X�V
  end
end
