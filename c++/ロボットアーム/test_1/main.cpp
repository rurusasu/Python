//----------------------------------------
//
// This program created from 'Win32 project'
//
//----------------------------------------
#include <Windows.h>
#include <stdio.h>

// ��������ADirectInput�ŕK�v�ȃR�[�h
#define DIRECTINPUT_VERSION    0x0800	// DirectInput�̃o�[�W�������
#include <dinput.h>

#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")
// �����܂ŁADirectInput�ŕK�v�ȃR�[�h

//----------------------------------------
// �萔
//----------------------------------------
#define APP_NAME      "DInputMouseTest"              // ���̃v���O�����̖��O
#define APP_TITLE     "DirectInput GameControl test" // ���̃v���O�����̃^�C�g��
#define SCREEN_WIDTH  (640)
#define SCREEN_HEIGHT (480)



//----------------------------------------
// �O���[�o���ϐ�
//----------------------------------------
// ��������ADirectInput�ɕK�v�ȃR�[�h
LPDIRECTINPUT8 g_pDInput = NULL; // DirectInput�I�u�W�F�N�g

