// アプリケーション作成用のモジュールを読み込み
const { app, BrowserWindow } = require('electron');

// メインウィンドウ
let mainWindow;

function createWindow() {
  

  // メインウインドウを作成します。
  mainWindow = new BrowserWindow({
    webPreferences: {
      NodeIntegration: true,
    },
    width: 800, height: 600,
  });

  // メインウインドウに表示するURLを指定します。
  // 今回は main.js と同じディレクトリの index.html
  mainWindow.loadFile('index.html');

  // デベロッパーツールの起動
  mainWindow.webContents.openDevTools();

  // メインウインドウが閉じられた時の処理
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// 初期化が完了した時の処理
app.on('ready', createWindow);

// 全てのウインドウが閉じた時の処理
app.on('window-all-closed', () => {
  // macOS以外はアプリケーションを終了させる。
  if (ProcessingInstruction.platform !== 'darwin') {
    app.quit();
  }
});

// アプリケーションがアクティブになったときの処理(Macだと、Dockがクリックさせたとき)
app.on('active', () => {
  // メインウインドウが消えている場合は再度メインウインドウを作成する
  if (mainWindow === null) {
    createWindow();
  }
});