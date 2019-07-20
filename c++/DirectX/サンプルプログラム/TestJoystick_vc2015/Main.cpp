#pragma warning( disable : 4996 )
#include "CWindow.h"
#include <stdio.h>

#define DIRECTINPUT_VERSION     0x0800          // DirectInputのバージョン指定
#include <dinput.h>

#define DEBUGMODE
#include "debug.h"

// ライブラリリンク
#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")


// グローバル変数定義
CWindow					win;					// ウインドウ
LPDIRECTINPUT8			lpDI = NULL;			// IDirectInput8
LPDIRECTINPUTDEVICE8	lpJoystick = NULL;		// ジョイスティックデバイス



// 1つのデバイスごとに呼び出されるコールバック関数
BOOL PASCAL EnumJoyDeviceProc( LPCDIDEVICEINSTANCE lpddi,LPVOID pvRef )
{
	DEBUG( "コールバック呼び出し\n" );

	// ジョイスティックデバイスの作成
	HRESULT ret = lpDI->CreateDevice( lpddi->guidInstance,&lpJoystick,NULL );
	if( FAILED(ret) ) {
		DEBUG( "デバイス作成失敗\n" );
		return DIENUM_STOP;
	}

	// 入力データ形式のセット
	ret = lpJoystick->SetDataFormat( &c_dfDIJoystick );
	if( FAILED(ret) ) {
		DEBUG( "入力データ形式のセット失敗\n" );
		lpJoystick->Release();
		return DIENUM_STOP;
	}

	// 排他制御のセット
	ret = lpJoystick->SetCooperativeLevel( win.hWnd,DISCL_FOREGROUND|DISCL_NONEXCLUSIVE|DISCL_NOWINKEY );
	if( FAILED(ret) ) {
		DEBUG( "排他制御のセット失敗\n" );
		lpJoystick->Release();
		return DIENUM_STOP;
	}

	// 入力範囲のセット
	DIPROPRANGE	diprg;
	diprg.diph.dwSize		= sizeof(diprg);
	diprg.diph.dwHeaderSize	= sizeof(diprg.diph);
	diprg.diph.dwHow		= DIPH_BYOFFSET;
	diprg.lMax				= 1000;
	diprg.lMin				= -1000;

	// X軸
	diprg.diph.dwObj = DIJOFS_X;
	lpJoystick->SetProperty( DIPROP_RANGE,&diprg.diph );

	// Y軸
	diprg.diph.dwObj = DIJOFS_Y;
	lpJoystick->SetProperty( DIPROP_RANGE,&diprg.diph );

	// Z軸
	diprg.diph.dwObj = DIJOFS_Z;
	lpJoystick->SetProperty( DIPROP_RANGE,&diprg.diph );

	// RX軸
	diprg.diph.dwObj = DIJOFS_RX;
	lpJoystick->SetProperty( DIPROP_RANGE,&diprg.diph );

	// RY軸
	diprg.diph.dwObj = DIJOFS_RY;
	lpJoystick->SetProperty( DIPROP_RANGE,&diprg.diph );

	// RZ軸
	diprg.diph.dwObj = DIJOFS_RZ;
	lpJoystick->SetProperty( DIPROP_RANGE,&diprg.diph );

	// 起動準備完了
	lpJoystick->Poll();

	// 構築完了なら
	char tmp[260];
	WideCharToMultiByte( CP_ACP,0,lpddi->tszInstanceName,-1,tmp,sizeof(tmp),NULL,NULL );
	DEBUG( "インスタンスの登録名 [%s]\n",tmp );
	WideCharToMultiByte( CP_ACP,0,lpddi->tszProductName,-1,tmp,sizeof(tmp),NULL,NULL );
	DEBUG( "製品の登録名         [%s]\n",tmp );
	DEBUG( "構築完了\n");

	// 最初の1つのみで終わる
	return DIENUM_STOP;			// 次のデバイスを列挙するにはDIENUM_CONTINUEを返す
}

// メインルーチン
int PASCAL WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdParam, int nCmdShow)
{
	INITDEBUG();
	CLEARDEBUG();

	// ウインドウ
	if( !win.Create(hInstance,L"TestJoystick") ) {
		DEBUG( "ウィンドウエラー\n" );
 		return -1;
	}

	// IDirectInput8の作成
	HRESULT ret = DirectInput8Create( hInstance,DIRECTINPUT_VERSION,IID_IDirectInput8,(LPVOID*)&lpDI,NULL );
	if( FAILED(ret) ) {
		// 作成に失敗
		DEBUG( "DirectInput8の作成に失敗\n" );
		return -1;
	}

	// ジョイスティックの列挙
	ret = lpDI->EnumDevices( DI8DEVCLASS_GAMECTRL,EnumJoyDeviceProc,NULL,DIEDFL_ATTACHEDONLY );
	if( FAILED(ret) ) {
		DEBUG( "ジョイスティックの列挙失敗\n" );
		lpDI->Release();
		return -1;
	}

	if( !lpJoystick ) {
		// ジョイスティックが1つも見つからない
		DEBUG( "ジョイスティックが1つも見つからない\n" );
		MessageBoxW( win.hWnd,L"ジョイスティックが1つも接続されていません",L"エラー",MB_OK|MB_ICONHAND );
		lpDI->Release();
		return -1;
	}

	// デバイス情報
	DIDEVCAPS dc;
	dc.dwSize = sizeof(dc);
	lpJoystick->GetCapabilities( &dc );
	DEBUG( "DIDC_ATTACHED           [%d]\n",(dc.dwFlags&DIDC_ATTACHED)?1:0 );
	DEBUG( "DIDC_POLLEDDEVICE       [%d]\n",(dc.dwFlags&DIDC_POLLEDDEVICE)?1:0 );
	DEBUG( "DIDC_EMULATED           [%d]\n",(dc.dwFlags&DIDC_EMULATED)?1:0 );
	DEBUG( "DIDC_FORCEFEEDBACK      [%d]\n",(dc.dwFlags&DIDC_FORCEFEEDBACK)?1:0 );
	DEBUG( "DIDC_FFATTACK           [%d]\n",(dc.dwFlags&DIDC_FFATTACK)?1:0 );
	DEBUG( "DIDC_FFFADE             [%d]\n",(dc.dwFlags&DIDC_FFFADE)?1:0 );
	DEBUG( "DIDC_SATURATION         [%d]\n",(dc.dwFlags&DIDC_SATURATION)?1:0 );
	DEBUG( "DIDC_POSNEGCOEFFICIENTS [%d]\n",(dc.dwFlags&DIDC_POSNEGCOEFFICIENTS)?1:0 );
	DEBUG( "DIDC_POSNEGSATURATION   [%d]\n",(dc.dwFlags&DIDC_POSNEGSATURATION)?1:0 );
	DEBUG( "DIDC_DEADBAND           [%d]\n",(dc.dwFlags&DIDC_DEADBAND)?1:0 );
	DEBUG( "DIDC_STARTDELAY         [%d]\n",(dc.dwFlags&DIDC_STARTDELAY)?1:0 );
	DEBUG( "DIDC_ALIAS              [%d]\n",(dc.dwFlags&DIDC_ALIAS)?1:0 );
	DEBUG( "DIDC_PHANTOM            [%d]\n",(dc.dwFlags&DIDC_PHANTOM)?1:0 );
	DEBUG( "DIDC_HIDDEN             [%d]\n",(dc.dwFlags&DIDC_HIDDEN)?1:0 );/**/

	// 動作開始
	lpJoystick->Acquire();

	// メインループ
	MSG msg;
	while(1) {
		if( PeekMessage(&msg,NULL,0,0,PM_REMOVE) ) {
			if( msg.message==WM_QUIT ) {
				DEBUG( "WM_QUIT\n" );
				break;
			}
			TranslateMessage(&msg);
        	DispatchMessage(&msg);
		}

		// データ取得前にPollが必要なら
		if( dc.dwFlags&DIDC_POLLEDDATAFORMAT ) {
			lpJoystick->Poll();
		}

		// ジョイスティックの入力
		DIJOYSTATE joy;
		ZeroMemory( &joy,sizeof(joy) );
		HRESULT ret = lpJoystick->GetDeviceState( sizeof(joy),&joy );
		if( FAILED(ret) ) {
			// 失敗なら再び動作開始を行う
			lpJoystick->Acquire();
		}

		// 入力状態表示
		HDC hdc = GetDC( win.hWnd );
		if( hdc ) {
			// 背景クリア
			BitBlt( hdc,0,0,640,480,NULL,0,0,WHITENESS );

			char s[256];
			sprintf( s,"X = %d",joy.lX );
			TextOutA( hdc,0,0,s,(int)strlen(s) );
			sprintf( s,"Y = %d",joy.lY );
			TextOutA( hdc,100,0,s,(int)strlen(s) );
			sprintf( s,"Z = %d",joy.lZ );
			TextOutA( hdc,200,0,s,(int)strlen(s) );

			sprintf( s,"RX = %d",joy.lRx );
			TextOutA( hdc,0,20,s,(int)strlen(s) );
			sprintf( s,"RY = %d",joy.lRy );
			TextOutA( hdc,100,20,s,(int)strlen(s) );
			sprintf( s,"RZ = %d",joy.lRz );
			TextOutA( hdc,200,20,s,(int)strlen(s) );

			sprintf( s,"Slider0 = %d",joy.rglSlider[0] );
			TextOutA( hdc,0,40,s,(int)strlen(s) );
			sprintf( s,"Slider1 = %d",joy.rglSlider[1] );
			TextOutA( hdc,100,40,s,(int)strlen(s) );

			sprintf( s,"POV0 = %d",joy.rgdwPOV[0] );
			TextOutA( hdc,0,60,s,(int)strlen(s) );
			sprintf( s,"POV1 = %d",joy.rgdwPOV[1] );
			TextOutA( hdc,100,60,s,(int)strlen(s) );
			sprintf( s,"POV2 = %d",joy.rgdwPOV[2] );
			TextOutA( hdc,200,60,s,(int)strlen(s) );
			sprintf( s,"POV3 = %d",joy.rgdwPOV[3] );
			TextOutA( hdc,300,60,s,(int)strlen(s) );

			int i;
			for( i=0;i<32;i++ ) {
				sprintf( s,"%d",(joy.rgbButtons[i]&0x80)?1:0 );
				TextOutA( hdc,i*10,80,s,(int)strlen(s) );
			}

			ReleaseDC( win.hWnd,hdc );
		}


		Sleep(8);										// CPUへのウエイト
	}

	lpJoystick->Release();
	lpDI->Release();

	win.Delete();

	return 0;
}
