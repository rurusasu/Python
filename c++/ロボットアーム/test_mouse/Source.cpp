//-----------------------------------------------------------------------------
//
// This program created from 'Win32 project'.
//
//-----------------------------------------------------------------------------
#include <windows.h>
#include <stdio.h>

// ��������ADirectInput�ŕK�v�ȃR�[�h -->
#define DIRECTINPUT_VERSION		0x0800		// DirectInput�̃o�[�W�������
#include <dinput.h>

#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")
// --> �����܂ŁADirectInput�ŕK�v�ȃR�[�h

//-----------------------------------------------------------------------------
// �萔
//-----------------------------------------------------------------------------
#define APP_NAME			"DInputMouseTest"			// ���̃v���O�����̖��O
#define APP_TITLE			"DirectInput Mouse test"	// ���̃v���O�����̃^�C�g��
#define SCREEN_WIDTH		(640)
#define SCREEN_HEIGHT		(480)


//-----------------------------------------------------------------------------
// �O���[�o���ϐ�
//-----------------------------------------------------------------------------
// ��������ADirectInput�ŕK�v�ȃR�[�h -->
LPDIRECTINPUT8       g_pDInput = NULL;	// DirectInput�I�u�W�F�N�g

// �}�E�X�p
LPDIRECTINPUTDEVICE8 g_pDIMouse = NULL;	// �}�E�X�f�o�C�X
DIMOUSESTATE g_zdiMouseState;			// �}�E�X���
// --> �����܂ŁADirectInput�ŕK�v�ȃR�[�h

BOOL g_bAppActive = FALSE;			// �A�v���P�[�V�����A�N�e�B�u�t���O

// ��������ADirectInput�ŕK�v�ȃR�[�h -->
//-----------------------------------------------------------------------------
//
// DirectInput�p�֐�
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// DirectInput�̏���������
//-----------------------------------------------------------------------------
bool InitDInput(HINSTANCE hInstApp, HWND hWnd)
{
	HRESULT ret = DirectInput8Create(hInstApp, DIRECTINPUT_VERSION, IID_IDirectInput8, (void**)& g_pDInput, NULL);
	if (FAILED(ret)) {
		return false;	// �쐬�Ɏ��s
	}
	return true;
}
//-----------------------------------------------------------------------------
// DirectInput�̏I������
//-----------------------------------------------------------------------------
bool ReleaseDInput(void)
{
	// DirectInput�̃f�o�C�X���J��
	if (g_pDInput) {
		g_pDInput->Release();
		g_pDInput = NULL;
	}

	return true;
}

//-----------------------------------------------------------------------------
//
// DirectInput(Mouse)�p�֐�
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// DirectInput�̃}�E�X�f�o�C�X�p�̏���������
//-----------------------------------------------------------------------------
bool InitDInputMouse(HWND hWnd)
{
	HRESULT ret = S_FALSE;
	if (g_pDInput == NULL) {
		return false;
	}

	// �}�E�X�p�Ƀf�o�C�X�I�u�W�F�N�g���쐬
	ret = g_pDInput->CreateDevice(GUID_SysMouse, &g_pDIMouse, NULL);
	if (FAILED(ret)) {
		// �f�o�C�X�̍쐬�Ɏ��s
		return false;
	}

	// �f�[�^�t�H�[�}�b�g��ݒ�
	ret = g_pDIMouse->SetDataFormat(&c_dfDIMouse);	// �}�E�X�p�̃f�[�^�E�t�H�[�}�b�g��ݒ�
	if (FAILED(ret)) {
		// �f�[�^�t�H�[�}�b�g�Ɏ��s
		return false;
	}

	// ���[�h��ݒ�i�t�H�A�O���E���h����r�����[�h�j
	ret = g_pDIMouse->SetCooperativeLevel(hWnd, DISCL_NONEXCLUSIVE | DISCL_FOREGROUND);
	if (FAILED(ret)) {
		// ���[�h�̐ݒ�Ɏ��s
		return false;
	}

	// �f�o�C�X�̐ݒ�
	DIPROPDWORD diprop;
	diprop.diph.dwSize = sizeof(diprop);
	diprop.diph.dwHeaderSize = sizeof(diprop.diph);
	diprop.diph.dwObj = 0;
	diprop.diph.dwHow = DIPH_DEVICE;
	diprop.dwData = DIPROPAXISMODE_REL;	// ���Βl���[�h�Őݒ�i��Βl��DIPROPAXISMODE_ABS�j

	ret = g_pDIMouse->SetProperty(DIPROP_AXISMODE, &diprop.diph);
	if (FAILED(ret)) {
		// �f�o�C�X�̐ݒ�Ɏ��s
		return false;
	}

	// ���͐���J�n
	g_pDIMouse->Acquire();

	return true;
}
//-----------------------------------------------------------------------------
// DirectInput�̃}�E�X�f�o�C�X�p�̉������
//-----------------------------------------------------------------------------
bool ReleaseDInputMouse()
{
	// DirectInput�̃f�o�C�X���J��
	if (g_pDIMouse) {
		g_pDIMouse->Release();
		g_pDIMouse = NULL;
	}

	return true;

}
// --> �����܂ŁADirectInput�ŕK�v�ȃR�[�h

//-----------------------------------------------------------------------------
// DirectInput�̃}�E�X�f�o�C�X��Ԏ擾����
//-----------------------------------------------------------------------------
void GetMouseState(HWND hWnd)
{
	if (g_pDIMouse == NULL) {
		// �I�u�W�F�N�g�����O�ɌĂ΂ꂽ�Ƃ��͂����Ő���������
		InitDInputMouse(hWnd);
	}

	// �ǎ�O�̒l��ێ����܂�
	DIMOUSESTATE g_zdiMouseState_bak;	// �}�E�X���(�ω����m�p)
	memcpy(&g_zdiMouseState_bak, &g_zdiMouseState, sizeof(g_zdiMouseState_bak));

	// ��������ADirectInput�ŕK�v�ȃR�[�h -->
		// �}�E�X�̏�Ԃ��擾���܂�
	HRESULT	hr = g_pDIMouse->GetDeviceState(sizeof(DIMOUSESTATE), &g_zdiMouseState);
	if (hr == DIERR_INPUTLOST) {
		g_pDIMouse->Acquire();
		hr = g_pDIMouse->GetDeviceState(sizeof(DIMOUSESTATE), &g_zdiMouseState);
	}
	// --> �����܂ŁADirectInput�ŕK�v�ȃR�[�h

	if (memcmp(&g_zdiMouseState_bak, &g_zdiMouseState, sizeof(g_zdiMouseState_bak)) != 0) {
		// �m�F�p�̏����A�������� -->
				// �l���ς������\�����܂�
		char buf[128];
		wsprintf(buf, "(%5d, %5d, %5d) %s %s %s\n",
			g_zdiMouseState.lX, g_zdiMouseState.lY, g_zdiMouseState.lZ,
			(g_zdiMouseState.rgbButtons[0] & 0x80) ? "Left" : "--",
			(g_zdiMouseState.rgbButtons[1] & 0x80) ? "Right" : "--",
			(g_zdiMouseState.rgbButtons[2] & 0x80) ? "Center" : "--");
		OutputDebugString(buf);
		// --> �����܂ŁA�m�F�p�̏���
	}
}


//-----------------------------------------------------------------------------
//
// Windows�A�v���P�[�V�����֐�
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// ���b�Z�[�W�n���h��
//-----------------------------------------------------------------------------
LRESULT CALLBACK WinProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_ACTIVATE:	// �A�N�e�B�u���F1�@��A�N�e�B�u���F0
		if (wParam == 0) {
			// ��A�N�e�B�u�ɂȂ����ꍇ
			ReleaseDInputMouse();
		}
		return 0L;
	case WM_DESTROY:	// �A�v���P�[�V�����I�����̏���
// ��������ADirectInput�ŕK�v�ȃR�[�h -->
		ReleaseDInputMouse();	// DirectInput(Mouse)�I�u�W�F�N�g�̊J��
		ReleaseDInput();		// DirectInput�I�u�W�F�N�g�̊J��
// --> �����܂ŁADirectInput�ŕK�v�ȃR�[�h
		// �v���O�������I�����܂�
		PostQuitMessage(0);
		return 0L;
	case WM_SETCURSOR:	// �J�[�\���̐ݒ�
		SetCursor(NULL);
		return TRUE;
	}

	return DefWindowProc(hWnd, message, wParam, lParam);
}

//-----------------------------------------------------------------------------
// �E�B���h�E������(����)����
//-----------------------------------------------------------------------------
HWND InitializeWindow(HINSTANCE hThisInst, int nWinMode)
{
	WNDCLASS wc;
	HWND     hWnd;		// Window Handle

	// �E�B���h�E�N���X���`����
	wc.hInstance = hThisInst;					// ���̃C���X�^���X�ւ̃n���h��
	wc.lpszClassName = APP_NAME;				// �E�B���h�E�N���X��
	wc.lpfnWndProc = WinProc;					// �E�B���h�E�֐�
	wc.style = CS_HREDRAW | CS_VREDRAW;			// �E�B���h�E�X�^�C��
	wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);	// �A�C�R��
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);	// �J�[�\���X�^�C��
	wc.lpszMenuName = APP_NAME;					// ���j���[�i�Ȃ��j
	wc.cbClsExtra = 0;							// �G�L�X�g���i�Ȃ��j
	wc.cbWndExtra = 0;							// �K�v�ȏ��i�Ȃ��j
	wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);	// �E�B���h�E�̔w�i�i���j

	// �E�B���h�E�N���X��o�^����
	if (!RegisterClass(&wc))
		return NULL;

	// �E�B���h�E�N���X�̓o�^���ł����̂ŁA�E�B���h�E�𐶐�����
	hWnd = CreateWindowEx(WS_EX_TOPMOST,
		APP_NAME,				// �E�B���h�E�N���X�̖��O
		APP_TITLE,				// �E�B���h�E�^�C�g��
		WS_OVERLAPPEDWINDOW,	// �E�B���h�E�X�^�C���i�m�[�}���j
		0,						// �E�B���h�E���p�w���W
		0,						// �E�B���h�E���p�x���W
		SCREEN_WIDTH,			// �E�B���h�E�̕�
		SCREEN_HEIGHT,			// �E�B���h�E�̍���
		NULL,					// �e�E�B���h�E�i�Ȃ��j
		NULL,					// ���j���[�i�Ȃ��j
		hThisInst,				// ���̃v���O�����̃C���X�^���X�̃n���h��
		NULL					// �ǉ������i�Ȃ��j
	);

	if (!hWnd) {
		return NULL;
	}

	// �E�B���h�E��\������
	ShowWindow(hWnd, nWinMode);
	UpdateWindow(hWnd);
	SetFocus(hWnd);

	return hWnd;
}

//-----------------------------------------------------------------------------
// �v���O�����G���g���[�|�C���g(WinMain)
//-----------------------------------------------------------------------------
int WINAPI WinMain(HINSTANCE hThisInst, HINSTANCE hPrevInst, LPSTR lpszArgs, int nWinMode)
{
	MSG  msg;
	HWND hWnd;

	/* �\������E�B���h�E�̒�`�A�o�^�A�\�� */
	if (!(hWnd = InitializeWindow(hThisInst, nWinMode))) {
		return FALSE;
	}

	// ��������ADirectInput�ŕK�v�ȃR�[�h -->
	InitDInput(hThisInst, hWnd);
	InitDInputMouse(hWnd);
	// --> �����܂ŁADirectInput�ŕK�v�ȃR�[�h

	while (TRUE)
	{
		if (PeekMessage(&msg, NULL, 0, 0, PM_NOREMOVE))
		{
			if (!GetMessage(&msg, NULL, 0, 0)) {
				break;
			}
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		// Check and get mouse state
// ��������ADirectInput�ŕK�v�ȃR�[�h -->
		GetMouseState(hWnd);
		// --> �����܂ŁADirectInput�ŕK�v�ȃR�[�h
	}

	return msg.wParam;
}