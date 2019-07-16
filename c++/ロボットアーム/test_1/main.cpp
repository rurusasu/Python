//----------------------------------------
//
// This program created from 'Win32 project'
//
//----------------------------------------
#include <Windows.h>
#include <stdio.h>

// ここから、DirectInputで必要なコード
#define DIRECTINPUT_VERSION    0x0800	// DirectInputのバージョン情報
#include <dinput.h>

#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")
// ここまで、DirectInputで必要なコード

//----------------------------------------
// 定数
//----------------------------------------
#define APP_NAME      "DInputMouseTest"              // このプログラムの名前
#define APP_TITLE     "DirectInput GameControl test" // このプログラムのタイトル
#define SCREEN_WIDTH  (640)
#define SCREEN_HEIGHT (480)



//----------------------------------------
// グローバル変数
//----------------------------------------
// ここから、DirectInputに必要なコード
LPDIRECTINPUT8 g_pDInput = NULL; // DirectInputオブジェクト

