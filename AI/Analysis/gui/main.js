const electron = require('electron');
const app = electron.app;
const BrowserWindow = electron.BrowserWindow;


let mainWindow = null;
app.on('ready', () => {
  // mainWindow を作成 (window の大きさや、kioskモードにするかどうかなどもここで定義できる)
  mainWindow = new BrowserWindow({ width: 400, height: 300 });
  // Electronに表示数rhtmlを絶対パスで指定(相対パスだと動かない)
    mainWindow.loadURL('file://' + __dirname + '/index.html');

  // ChromiumのDevツールを開く
  mainWindow.webContents.openDevTools();

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
});