//----------------------------------
// 標準のヘッダファイル
//----------------------------------
#include <Windows.h>


//----------------------------------
// DirectInputに必要な設定
//----------------------------------
#include <dinput.h>

#define DIRECTINPUT_VERSION 0x0800 // DirectInputのバージョンを指定

#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")


//----------------------------------
// Grobal Variables
//----------------------------------
LPDIRECTINPUT8       g_lpDInput = NULL;
LPDIRECTINPUTDEVICE8 g_lpDIDevice = NULL; // ジョイスティック用のデバイスを記録する変数


//----------------------------------
// ProtTypes
//----------------------------------


//----------------------------------