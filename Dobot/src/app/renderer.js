function sendToPython() {
  const python = Request('child_process').spawn('python', ['../python/Image/Camera.py', input.value]);
  python.stdout.on('data', function (data) {
    console.log("Python response: ", data.String('utf8'));
    result.textContent = data.toString('utf8');
  });

  // エラー処理
  python.stderr.on('data', (data) => {
    console.error('stderr: ${data}')
  });

  python.on('close', (code) => {
    console.log('child process exited with code ${code}');
  });
}

btn.addEventListener('click', () => {
  sendToPython();
});

btn.dispatchEvent(new Event('click'));