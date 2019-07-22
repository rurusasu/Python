// DXlib_test_3.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include <iostream>
#include <DxLib.h>

int main()
{
	DINPUT_JOYSTATE g_pDInput;
	int i;

	// DXライブラリの初期化
	if (DxLib_Init() < 0) return -1;

	// メインループ
	// ESCキーが押されるまでループ
	while ((GetJoypadInputState(DX_INPUT_PAD1) & PAD_INPUT_9) == 0)
	{
		// 入力状態を取得
		GetJoypadDirectInputState(DX_INPUT_PAD1, &g_pDInput);

		// 画面に表示
		printf("X:%d Y:%d Z:%d", g_pDInput.X, g_pDInput.Y, g_pDInput.Z);
		printf("Rx:%d Ry:%d Rz:%d", g_pDInput.Rx, g_pDInput.Ry, g_pDInput.Rz);
		printf("POV 0:%d", g_pDInput.POV[0]);
		for (i = 0; i < 32; i++)
		{
			printf("%2d:%d", i, g_pDInput.Buttons[i]);
		}
	}

	// DXライブラリの後始末
	DxLib_End();

	// ソフトの終了
	return 0;
}

// プログラムの実行: Ctrl + F5 または [デバッグ] > [デバッグなしで開始] メニュー
// プログラムのデバッグ: F5 または [デバッグ] > [デバッグの開始] メニュー

// 作業を開始するためのヒント: 
//    1. ソリューション エクスプローラー ウィンドウを使用してファイルを追加/管理します 
//   2. チーム エクスプローラー ウィンドウを使用してソース管理に接続します
//   3. 出力ウィンドウを使用して、ビルド出力とその他のメッセージを表示します
//   4. エラー一覧ウィンドウを使用してエラーを表示します
//   5. [プロジェクト] > [新しい項目の追加] と移動して新しいコード ファイルを作成するか、[プロジェクト] > [既存の項目の追加] と移動して既存のコード ファイルをプロジェクトに追加します
//   6. 後ほどこのプロジェクトを再び開く場合、[ファイル] > [開く] > [プロジェクト] と移動して .sln ファイルを選択します
