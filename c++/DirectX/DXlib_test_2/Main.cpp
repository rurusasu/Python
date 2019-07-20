#include "DxLib.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	DINPUT_JOYSTATE input;
	int i;
	int Color;

	// �c�w���C�u�����̏�����
	if (DxLib_Init() < 0) return -1;

	// �`���𗠉�ʂɂ���
	SetDrawScreen(DX_SCREEN_BACK);

	// ���C�����[�v(�����L�[�������ꂽ�烋�[�v�𔲂���)
	while (ProcessMessage() == 0)
	{
		// ��ʂ̃N���A
		ClearDrawScreen();

		// ���͏�Ԃ��擾
		GetJoypadDirectInputState(DX_INPUT_PAD1, &input);

		// ��ʂɍ\���̂̒��g��`��
		Color = GetColor(255, 255, 255);
		DrawFormatString(0, 0, Color, "X:%d Y:%d Z:%d",
			input.X, input.Y, input.Z);
		DrawFormatString(0, 16, Color, "Rx:%d Ry:%d Rz:%d",
			input.Rx, input.Ry, input.Rz);
		DrawFormatString(0, 32, Color, "Slider 0:%d 1:%d",
			input.Slider[0], input.Slider[1]);
		DrawFormatString(0, 48, Color, "POV 0:%d 1:%d 2:%d 3:%d",
			input.POV[0], input.POV[1],
			input.POV[2], input.POV[3]);
		DrawString(0, 64, "Button", Color);
		for (i = 0; i < 32; i++)
		{
			DrawFormatString(64 + i % 8 * 64, 64 + i / 8 * 16, Color,
				"%2d:%d", i, input.Buttons[i]);
		}

		// ����ʂ̓��e��\��ʂɔ��f
		ScreenFlip();
	}

	// �c�w���C�u�����̌�n��
	DxLib_End();

	// �\�t�g�̏I��
	return 0;
}