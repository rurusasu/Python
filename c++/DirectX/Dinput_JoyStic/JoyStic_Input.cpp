//----------------------------------
// �W���̃w�b�_�t�@�C��
//----------------------------------
#include <Windows.h>


//----------------------------------
// DirectInput�ɕK�v�Ȑݒ�
//----------------------------------
#include <dinput.h>

#define DIRECTINPUT_VERSION 0x0800 // DirectInput�̃o�[�W�������w��

#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")


//----------------------------------
// Grobal Variables
//----------------------------------
LPDIRECTINPUT8       g_lpDInput = NULL;
LPDIRECTINPUTDEVICE8 g_lpDIDevice = NULL; // �W���C�X�e�B�b�N�p�̃f�o�C�X���L�^����ϐ�


//----------------------------------
// ProtTypes
//----------------------------------


//----------------------------------