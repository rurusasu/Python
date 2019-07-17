//-----------------------------------------------------------------------------
//
// This program created from 'Win32 project'.
//
//-----------------------------------------------------------------------------
#include <windows.h>
#include <stdio.h>

// ここから、DirectInputで必要なコード -->
#define DIRECTINPUT_VERSION		0x0800		// DirectInputのバージョン情報
#include <dinput.h>

#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")
// --> ここまで、DirectInputで必要なコード

//-----------------------------------------------------------------------------
// 定数
//-----------------------------------------------------------------------------
#define APP_NAME			"DInputMouseTest"			// このプログラムの名前
#define APP_TITLE			"DirectInput Mouse test"	// このプログラムのタイトル
#define SCREEN_WIDTH		(640)
#define SCREEN_HEIGHT		(480)


//-----------------------------------------------------------------------------
// グローバル変数
//-----------------------------------------------------------------------------
// ここから、DirectInputで必要なコード -->
LPDIRECTINPUT8       g_pDInput = NULL;	// DirectInputオブジェクト

// マウス用
LPDIRECTINPUTDEVICE8 g_pDIMouse = NULL;	// マウスデバイス
DIMOUSESTATE g_zdiMouseState;			// マウス状態
// --> ここまで、DirectInputで必要なコード

BOOL g_bAppActive = FALSE;			// アプリケーションアクティブフラグ

// ここから、DirectInputで必要なコード -->
//-----------------------------------------------------------------------------
//
// DirectInput用関数
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// DirectInputの初期化処理
//-----------------------------------------------------------------------------
bool InitDInput(HINSTANCE hInstApp, HWND hWnd)
{
	HRESULT ret = DirectInput8Create(hInstApp, DIRECTINPUT_VERSION, IID_IDirectInput8, (void**)& g_pDInput, NULL);
	if (FAILED(ret)) {
		return false;	// 作成に失敗
	}
	return true;
}
//-----------------------------------------------------------------------------
// DirectInputの終了処理
//-----------------------------------------------------------------------------
bool ReleaseDInput(void)
{
	// DirectInputのデバイスを開放
	if (g_pDInput) {
		g_pDInput->Release();
		g_pDInput = NULL;
	}

	return true;
}

//-----------------------------------------------------------------------------
//
// DirectInput(Mouse)用関数
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// DirectInputのマウスデバイス用の初期化処理
//-----------------------------------------------------------------------------
bool InitDInputMouse(HWND hWnd)
{
	HRESULT ret = S_FALSE;
	if (g_pDInput == NULL) {
		return false;
	}

	// マウス用にデバイスオブジェクトを作成
	ret = g_pDInput->CreateDevice(GUID_SysMouse, &g_pDIMouse, NULL);
	if (FAILED(ret)) {
		// デバイスの作成に失敗
		return false;
	}

	// データフォーマットを設定
	ret = g_pDIMouse->SetDataFormat(&c_dfDIMouse);	// マウス用のデータ・フォーマットを設定
	if (FAILED(ret)) {
		// データフォーマットに失敗
		return false;
	}

	// モードを設定（フォアグラウンド＆非排他モード）
	ret = g_pDIMouse->SetCooperativeLevel(hWnd, DISCL_NONEXCLUSIVE | DISCL_FOREGROUND);
	if (FAILED(ret)) {
		// モードの設定に失敗
		return false;
	}

	// デバイスの設定
	DIPROPDWORD diprop;
	diprop.diph.dwSize = sizeof(diprop);
	diprop.diph.dwHeaderSize = sizeof(diprop.diph);
	diprop.diph.dwObj = 0;
	diprop.diph.dwHow = DIPH_DEVICE;
	diprop.dwData = DIPROPAXISMODE_REL;	// 相対値モードで設定（絶対値はDIPROPAXISMODE_ABS）

	ret = g_pDIMouse->SetProperty(DIPROP_AXISMODE, &diprop.diph);
	if (FAILED(ret)) {
		// デバイスの設定に失敗
		return false;
	}

	// 入力制御開始
	g_pDIMouse->Acquire();

	return true;
}
//-----------------------------------------------------------------------------
// DirectInputのマウスデバイス用の解放処理
//-----------------------------------------------------------------------------
bool ReleaseDInputMouse()
{
	// DirectInputのデバイスを開放
	if (g_pDIMouse) {
		g_pDIMouse->Release();
		g_pDIMouse = NULL;
	}

	return true;

}
// --> ここまで、DirectInputで必要なコード

//-----------------------------------------------------------------------------
// DirectInputのマウスデバイス状態取得処理
//-----------------------------------------------------------------------------
void GetMouseState(HWND hWnd)
{
	if (g_pDIMouse == NULL) {
		// オブジェクト生成前に呼ばれたときはここで生成させる
		InitDInputMouse(hWnd);
	}

	// 読取前の値を保持します
	DIMOUSESTATE g_zdiMouseState_bak;	// マウス情報(変化検知用)
	memcpy(&g_zdiMouseState_bak, &g_zdiMouseState, sizeof(g_zdiMouseState_bak));

	// ここから、DirectInputで必要なコード -->
		// マウスの状態を取得します
	HRESULT	hr = g_pDIMouse->GetDeviceState(sizeof(DIMOUSESTATE), &g_zdiMouseState);
	if (hr == DIERR_INPUTLOST) {
		g_pDIMouse->Acquire();
		hr = g_pDIMouse->GetDeviceState(sizeof(DIMOUSESTATE), &g_zdiMouseState);
	}
	// --> ここまで、DirectInputで必要なコード

	if (memcmp(&g_zdiMouseState_bak, &g_zdiMouseState, sizeof(g_zdiMouseState_bak)) != 0) {
		// 確認用の処理、ここから -->
				// 値が変わったら表示します
		char buf[128];
		wsprintf(buf, "(%5d, %5d, %5d) %s %s %s\n",
			g_zdiMouseState.lX, g_zdiMouseState.lY, g_zdiMouseState.lZ,
			(g_zdiMouseState.rgbButtons[0] & 0x80) ? "Left" : "--",
			(g_zdiMouseState.rgbButtons[1] & 0x80) ? "Right" : "--",
			(g_zdiMouseState.rgbButtons[2] & 0x80) ? "Center" : "--");
		OutputDebugString(buf);
		// --> ここまで、確認用の処理
	}
}


//-----------------------------------------------------------------------------
//
// Windowsアプリケーション関数
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// メッセージハンドラ
//-----------------------------------------------------------------------------
LRESULT CALLBACK WinProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_ACTIVATE:	// アクティブ時：1　非アクティブ時：0
		if (wParam == 0) {
			// 非アクティブになった場合
			ReleaseDInputMouse();
		}
		return 0L;
	case WM_DESTROY:	// アプリケーション終了時の処理
// ここから、DirectInputで必要なコード -->
		ReleaseDInputMouse();	// DirectInput(Mouse)オブジェクトの開放
		ReleaseDInput();		// DirectInputオブジェクトの開放
// --> ここまで、DirectInputで必要なコード
		// プログラムを終了します
		PostQuitMessage(0);
		return 0L;
	case WM_SETCURSOR:	// カーソルの設定
		SetCursor(NULL);
		return TRUE;
	}

	return DefWindowProc(hWnd, message, wParam, lParam);
}

//-----------------------------------------------------------------------------
// ウィンドウ初期化(生成)処理
//-----------------------------------------------------------------------------
HWND InitializeWindow(HINSTANCE hThisInst, int nWinMode)
{
	WNDCLASS wc;
	HWND     hWnd;		// Window Handle

	// ウィンドウクラスを定義する
	wc.hInstance = hThisInst;					// このインスタンスへのハンドル
	wc.lpszClassName = APP_NAME;				// ウィンドウクラス名
	wc.lpfnWndProc = WinProc;					// ウィンドウ関数
	wc.style = CS_HREDRAW | CS_VREDRAW;			// ウィンドウスタイル
	wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);	// アイコン
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);	// カーソルスタイル
	wc.lpszMenuName = APP_NAME;					// メニュー（なし）
	wc.cbClsExtra = 0;							// エキストラ（なし）
	wc.cbWndExtra = 0;							// 必要な情報（なし）
	wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);	// ウィンドウの背景（黒）

	// ウィンドウクラスを登録する
	if (!RegisterClass(&wc))
		return NULL;

	// ウィンドウクラスの登録ができたので、ウィンドウを生成する
	hWnd = CreateWindowEx(WS_EX_TOPMOST,
		APP_NAME,				// ウィンドウクラスの名前
		APP_TITLE,				// ウィンドウタイトル
		WS_OVERLAPPEDWINDOW,	// ウィンドウスタイル（ノーマル）
		0,						// ウィンドウ左角Ｘ座標
		0,						// ウィンドウ左角Ｙ座標
		SCREEN_WIDTH,			// ウィンドウの幅
		SCREEN_HEIGHT,			// ウィンドウの高さ
		NULL,					// 親ウィンドウ（なし）
		NULL,					// メニュー（なし）
		hThisInst,				// このプログラムのインスタンスのハンドル
		NULL					// 追加引数（なし）
	);

	if (!hWnd) {
		return NULL;
	}

	// ウィンドウを表示する
	ShowWindow(hWnd, nWinMode);
	UpdateWindow(hWnd);
	SetFocus(hWnd);

	return hWnd;
}

//-----------------------------------------------------------------------------
// プログラムエントリーポイント(WinMain)
//-----------------------------------------------------------------------------
int WINAPI WinMain(HINSTANCE hThisInst, HINSTANCE hPrevInst, LPSTR lpszArgs, int nWinMode)
{
	MSG  msg;
	HWND hWnd;

	/* 表示するウィンドウの定義、登録、表示 */
	if (!(hWnd = InitializeWindow(hThisInst, nWinMode))) {
		return FALSE;
	}

	// ここから、DirectInputで必要なコード -->
	InitDInput(hThisInst, hWnd);
	InitDInputMouse(hWnd);
	// --> ここまで、DirectInputで必要なコード

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
// ここから、DirectInputで必要なコード -->
		GetMouseState(hWnd);
		// --> ここまで、DirectInputで必要なコード
	}

	return msg.wParam;
}