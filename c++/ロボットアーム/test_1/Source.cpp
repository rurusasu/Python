//-----------------------------------------------------------------
//
//    DirectInput Sample Program.
//
//-----------------------------------------------------------------
#define INITGUID

#include <stdio.h>
#include <windows.h>
#include <dinput.h>
//#include <dinputex.h>

#pragma comment( lib , "dinput8.dll" )



//-----------------------------------------------------------------
//    Grobal Variables.
//-----------------------------------------------------------------
LPDIRECTINPUT8       g_lpDI;
LPDIRECTINPUTDEVICE8 g_lpDIDevice;
DIDEVCAPS            g_diDevCaps;

HWND                 g_hWnd;



//-----------------------------------------------------------------
//    Prototypes.
//-----------------------------------------------------------------
HWND    InitApp(HINSTANCE, int);
BOOL    InitDirectInput(HWND);
BOOL    ReadInput();
BOOL    CleanupDirectInput();
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
BOOL    CALLBACK EnumJoysticksCallback(const DIDEVICEINSTANCE*, VOID*);
BOOL    CALLBACK EnumAxesCallback(const DIDEVICEOBJECTINSTANCE*, VOID*);



//-----------------------------------------------------------------
//    Main.
//-----------------------------------------------------------------
int WINAPI WinMain(_In_ HINSTANCE hInst, _In_opt_ HINSTANCE hPrevinst, _In_ LPSTR lpCmdLine, _In_ int nCmdShow)
{
	MSG msg;

	g_hWnd = InitApp(hInst, nCmdShow);
	if (!g_hWnd) return FALSE;

	if (!InitDirectInput(g_hWnd)) return FALSE;

	while (msg.message != WM_QUIT) {
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else {
			ReadInput();
		}
		Sleep(1);
	}

	return msg.wParam;
}



//-----------------------------------------------------------------
//    Initialize Application.
//-----------------------------------------------------------------
HWND InitApp(HINSTANCE hInst, int nCmdShow)
{
	WNDCLASS wc;
	HWND hWnd;
	char szClassName[] = "DirectInput Test";

	wc.style = CS_HREDRAW | CS_VREDRAW;
	wc.hInstance = hInst;
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	wc.lpszClassName = szClassName;
	wc.lpszMenuName = NULL;
	wc.lpfnWndProc = WndProc;
	wc.cbWndExtra = 0;
	wc.cbClsExtra = 0;
	if (!RegisterClass(&wc)) return FALSE;

	hWnd = CreateWindow(szClassName, "Direct3D Test", WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, CW_USEDEFAULT, 640, 480,
		NULL, NULL, hInst, NULL);
	if (!hWnd) return FALSE;

	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);

	return hWnd;
}



//-----------------------------------------------------------------
//    Initialize DirectInput.
//-----------------------------------------------------------------
BOOL InitDirectInput(HWND hWnd)
{
	HINSTANCE hInst;
	HRESULT   hr;
	
	hInst = (HINSTANCE)GetWindowLong(hWnd, GWL_HINSTANCE);

	hr = DirectInput8Create(hInst, DIRECTINPUT_VERSION, IID_IDirectInput8,
		(void**)& g_lpDI, NULL);
	if (FAILED(hr)) {
		MessageBox(hWnd, "Can't create DirectInput object.", "Error", MB_OK);
		return FALSE;
	}

	hr = g_lpDI->EnumDevices(DI8DEVCLASS_GAMECTRL, EnumJoysticksCallback,
		NULL, DIEDFL_ATTACHEDONLY);
	if (FAILED(hr)) {
		MessageBox(hWnd, "Can't create Device.", "Error", MB_OK);
		return FALSE;
	}

	hr = g_lpDIDevice->SetDataFormat(&c_dfDIJoystick);
	if (FAILED(hr)) {
		MessageBox(hWnd, "Can't set data format.", "Error", MB_OK);
		return FALSE;
	}

	hr = g_lpDIDevice->SetCooperativeLevel(hWnd, DISCL_EXCLUSIVE | DISCL_FOREGROUND);
	if (FAILED(hr)) {
		MessageBox(hWnd, "Can't set cooperative level.", "Error", MB_OK);
		return FALSE;
	}

	g_diDevCaps.dwSize = sizeof(DIDEVCAPS);
	hr = g_lpDIDevice->GetCapabilities(&g_diDevCaps);
	if (FAILED(hr)) {
		MessageBox(hWnd, "Can't create device capabilities.", "Error", MB_OK);
		return FALSE;
	}

	hr = g_lpDIDevice->EnumObjects(EnumAxesCallback, (VOID*)hWnd, DIDFT_AXIS);
	if (FAILED(hr)) {
		MessageBox(hWnd, "Can't set property.", "Error", MB_OK);
		return FALSE;
	}

	hr = g_lpDIDevice->Poll();
	if (FAILED(hr)) {
		hr = g_lpDIDevice->Acquire();
		while (hr == DIERR_INPUTLOST) {
			hr = g_lpDIDevice->Acquire();
		}
	}

	return TRUE;
}



//------------------------------------------------------------------------------
//    Joysticks Callback.
//------------------------------------------------------------------------------
BOOL CALLBACK EnumJoysticksCallback(const DIDEVICEINSTANCE* pdidInstance, VOID* pContext)
{
	HRESULT hr;

	hr = g_lpDI->CreateDevice(pdidInstance->guidInstance, &g_lpDIDevice, NULL);
	
	if (FAILED(hr)) return DIENUM_CONTINUE;

	return DIENUM_STOP;
}



//------------------------------------------------------------------------------
//    Axes Callback.
//------------------------------------------------------------------------------
BOOL CALLBACK EnumAxesCallback(const DIDEVICEOBJECTINSTANCE* pdidoi, VOID* pContext)
{
	HRESULT     hr;
	DIPROPRANGE diprg;

	diprg.diph.dwSize = sizeof(DIPROPRANGE);
	diprg.diph.dwHeaderSize = sizeof(DIPROPHEADER);
	diprg.diph.dwHow = DIPH_BYID;
	diprg.diph.dwObj = pdidoi->dwType;
	diprg.lMin = 0 - 1000;
	diprg.lMax = 0 + 1000;
	hr = g_lpDIDevice->SetProperty(DIPROP_RANGE, &diprg.diph);

	if (FAILED(hr)) return DIENUM_STOP;

	return DIENUM_CONTINUE;
}



//-----------------------------------------------------------------
//    Cleanup DirectInput.
//-----------------------------------------------------------------
BOOL CleanupDirectInput()
{
	g_lpDIDevice->Unacquire();

	if (g_lpDIDevice != NULL)
		g_lpDIDevice->Release();

	if (g_lpDI != NULL)
		g_lpDI->Release();

	return TRUE;
}



//-----------------------------------------------------------------
//    Window Proc.
//-----------------------------------------------------------------
LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wp, LPARAM lp)
{
	switch (msg) {
	case WM_DESTROY:
		CleanupDirectInput();
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hWnd, msg, wp, lp);
	}

	return 0L;
}



//-----------------------------------------------------------------
//    Read Input.
//-----------------------------------------------------------------
BOOL ReadInput()
{
	DIJOYSTATE js;
	HRESULT    hr;
	int        i;
	char       titlebar[32];
	char       subbuf[4];

	if (NULL == g_lpDIDevice) return FALSE;

	hr = g_lpDIDevice->Poll();
	if (FAILED(hr)) return FALSE;

	hr = g_lpDIDevice->GetDeviceState(sizeof(DIJOYSTATE), &js);
	if (FAILED(hr)) return FALSE;

	titlebar[0] = '\0';

	if (js.lY == -1000) strcat_s(titlebar, sizeof(titlebar), "Å™");
	if (js.lY == 1000) strcat_s(titlebar, sizeof(titlebar), "Å´");
	if (js.lX == -1000) strcat_s(titlebar, sizeof(titlebar), "Å©");
	if (js.lX == 1000) strcat_s(titlebar, sizeof(titlebar), "Å®");

	for (i = 0; i < 32; i++) {
		if (js.rgbButtons[i] & 0x80) {
			sprintf_s(subbuf, ",%d", i);
			strcat_s(titlebar, sizeof(titlebar), subbuf);
		}
	}

	SetWindowText(g_hWnd, titlebar);

	return TRUE;
}