// DobotDemoDlg.cpp : implementation file
//

#include "stdafx.h"
#include "DobotDemo.h"
#include "DobotDemoDlg.h"
#include <conio.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#define ENABLE_CONSOLE_PRINT 1

float laserHeight;
// CDobotDemoDlg dialog

// Dobot specified library and header

#pragma comment(lib, "./DobotDll/DobotDll.lib")

#include "./DobotDll/DobotDll.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


CDobotDemoDlg::CDobotDemoDlg(CWnd* pParent /*=NULL*/)
    : CDialog(CDobotDemoDlg::IDD, pParent)
	, m_velocity(0)
	, m_acceleration(0)
//	, m_velocity2(_T(""))
//	, m_acceleration2(_T(""))
, m_velocity2(0)
, m_acceleration2(0)
{
    m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

    // Private member
    m_bConnectStatus = false;
}

void CDobotDemoDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialog::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_BUTTON_CONNECT, m_ConnectButton);
	DDX_Control(pDX, IDC_COMBO_JOGMODE, m_JOGMode);
	DDX_Control(pDX, IDC_BUTTON_J1P, m_ButtonJ1P);
	DDX_Control(pDX, IDC_BUTTON_J1N, m_ButtonJ1N);
	DDX_Control(pDX, IDC_BUTTON_J2P, m_ButtonJ2P);
	DDX_Control(pDX, IDC_BUTTON_J2N, m_ButtonJ2N);
	DDX_Control(pDX, IDC_BUTTON_J3P, m_ButtonJ3P);
	DDX_Control(pDX, IDC_BUTTON_J3N, m_ButtonJ3N);
	DDX_Control(pDX, IDC_BUTTON_J4P, m_ButtonJ4P);
	DDX_Control(pDX, IDC_BUTTON_J4N, m_ButtonJ4N);
	DDX_Control(pDX, IDC_STATIC_J1, m_StaticJ1);
	DDX_Control(pDX, IDC_STATIC_J2, m_StaticJ2);
	DDX_Control(pDX, IDC_STATIC_J3, m_StaticJ3);
	DDX_Control(pDX, IDC_STATIC_J4, m_StaticJ4);
	DDX_Control(pDX, IDC_STATIC_X, m_StaticX);
	DDX_Control(pDX, IDC_STATIC_Y, m_StaticY);
	DDX_Control(pDX, IDC_STATIC_Z, m_StaticZ);
	DDX_Control(pDX, IDC_STATIC_R, m_StaticR);
	DDX_Control(pDX, IDC_EDIT_X, m_EditX);
	DDX_Control(pDX, IDC_EDIT_Y, m_EditY);
	DDX_Control(pDX, IDC_EDIT_Z, m_EditZ);
	DDX_Control(pDX, IDC_EDIT_R, m_EditR);
	DDX_Control(pDX, IDC_BUTTON_SENDPTP, m_ButtonSendPTP);
	DDX_Control(pDX, IDC_EDIT1, m_edit);
	DDX_Control(pDX, IDC_EDIT2, m_edit2);
	DDX_Control(pDX, IDC_EDIT3, m_edit3);
	DDX_Control(pDX, IDC_EDIT4, m_edit4);
	DDX_Control(pDX, IDC_EDIT5, m_edit5);
	DDX_Text(pDX, IDC_EDIT6, m_velocity);
	DDV_MinMaxInt(pDX, m_velocity, 1, 1000);
	DDX_Text(pDX, IDC_EDIT7, m_acceleration);
	DDV_MinMaxInt(pDX, m_acceleration, 1, 1000);
	DDX_Control(pDX, IDC_EDIT6, m_edit6);
	DDX_Control(pDX, IDC_EDIT7, m_edit7);
	//	DDX_Text(pDX, IDC_EDIT8, m_velocity2);
	//	DDX_Text(pDX, IDC_EDIT9, m_acceleration2);
	DDX_Text(pDX, IDC_EDIT8, m_velocity2);
	DDX_Text(pDX, IDC_EDIT9, m_acceleration2);
	DDX_Control(pDX, IDC_EDIT8, m_velocity3);
	DDX_Control(pDX, IDC_EDIT9, m_acceleration3);
	DDX_Control(pDX, IDC_EDIT10, m_edit10);
}

BEGIN_MESSAGE_MAP(CDobotDemoDlg, CDialog)
    ON_WM_PAINT()
    ON_WM_QUERYDRAGICON()
    //}}AFX_MSG_MAP
    ON_BN_CLICKED(IDC_BUTTON_CONNECT, &CDobotDemoDlg::OnBnClickedButtonConnect)
    ON_WM_TIMER()
    ON_CBN_SELCHANGE(IDC_COMBO_JOGMODE, &CDobotDemoDlg::OnCbnSelchangeComboJOGMode)
    ON_BN_CLICKED(IDC_BUTTON_SENDPTP, &CDobotDemoDlg::OnBnClickedButtonSendPTP)
	ON_BN_CLICKED(IDC_BUTTON1, &CDobotDemoDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CDobotDemoDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CDobotDemoDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CDobotDemoDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CDobotDemoDlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CDobotDemoDlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &CDobotDemoDlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &CDobotDemoDlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON10, &CDobotDemoDlg::OnBnClickedButton10)
	ON_BN_CLICKED(IDC_BUTTON11, &CDobotDemoDlg::OnBnClickedButton11)
	ON_BN_CLICKED(IDC_BUTTON12, &CDobotDemoDlg::OnBnClickedButton12)
	ON_BN_CLICKED(IDC_BUTTON13, &CDobotDemoDlg::OnBnClickedButton13)
	ON_BN_CLICKED(IDC_BUTTON14, &CDobotDemoDlg::OnBnClickedButton14)
	ON_BN_CLICKED(IDC_BUTTON15, &CDobotDemoDlg::OnBnClickedButton15)
	ON_BN_CLICKED(IDC_BUTTON16, &CDobotDemoDlg::OnBnClickedButton16)
END_MESSAGE_MAP()


// CDobotDemoDlg message handlers
// 初期化ルーチン。起動時に１回だけ実行される。
BOOL CDobotDemoDlg::OnInitDialog()
{
    CDialog::OnInitDialog();

    // Set the icon for this dialog.  The framework does this automatically
    //  when the application's main window is not a dialog
    SetIcon(m_hIcon, TRUE);            // Set big icon
    SetIcon(m_hIcon, FALSE);        // Set small icon

    // TODO: Add extra initialization here
#if ENABLE_CONSOLE_PRINT
    AllocConsole();
#endif
    InitControls();

//	SetTimer(1, 100, NULL); // タイマを起動する。 // 引数の意味（タイマーID、呼び出す間隔(ms), NULLでOK） 
	CString sss;
	m_velocity = 200;
	sss.Format("%d", m_velocity);
	m_edit6.SetWindowText(sss);

	m_acceleration = 200;
	sss.Format("%d", m_velocity);
	m_edit7.SetWindowText(sss);



    return TRUE;  // return TRUE  unless you set the focus to a control
}


void CDobotDemoDlg::InitControls(void)
{
    m_JOGMode.InsertString(0, "Joint");
    m_JOGMode.InsertString(1, "Coordinate");
    m_JOGMode.SetCurSel(0);
    OnCbnSelchangeComboJOGMode();
    RefreshButtons();


}

void CDobotDemoDlg::OnCbnSelchangeComboJOGMode()
{
    // TODO: Add your control notification handler code here
    int currentSelect = m_JOGMode.GetCurSel();

    if (currentSelect == 0) {
        m_ButtonJ1P.SetWindowText("J1+");
        m_ButtonJ1N.SetWindowText("J1-");
        m_ButtonJ2P.SetWindowText("J2+");
        m_ButtonJ2N.SetWindowText("J2-");
        m_ButtonJ3P.SetWindowText("J3+");
        m_ButtonJ3N.SetWindowText("J3-");
        m_ButtonJ4P.SetWindowText("J4+");
        m_ButtonJ4N.SetWindowText("J4-");
    } else {
        m_ButtonJ1P.SetWindowText("X+");
        m_ButtonJ1N.SetWindowText("X-");
        m_ButtonJ2P.SetWindowText("Y+");
        m_ButtonJ2N.SetWindowText("Y-");
        m_ButtonJ3P.SetWindowText("Z+");
        m_ButtonJ3N.SetWindowText("Z-");
        m_ButtonJ4P.SetWindowText("R+");
        m_ButtonJ4N.SetWindowText("R-");
    }
}

void CDobotDemoDlg::RefreshButtons(void)
{
    if (m_bConnectStatus) {
        m_ConnectButton.SetWindowText("Disconnect");
    } else {
        m_ConnectButton.SetWindowText("Connect");
    }
    m_ButtonJ1P.EnableWindow(m_bConnectStatus);
    m_ButtonJ1N.EnableWindow(m_bConnectStatus);
    m_ButtonJ2P.EnableWindow(m_bConnectStatus);
    m_ButtonJ2N.EnableWindow(m_bConnectStatus);
    m_ButtonJ3P.EnableWindow(m_bConnectStatus);
    m_ButtonJ3N.EnableWindow(m_bConnectStatus);
    m_ButtonJ4P.EnableWindow(m_bConnectStatus);
    m_ButtonJ4N.EnableWindow(m_bConnectStatus);
    m_JOGMode.EnableWindow(m_bConnectStatus);

    m_EditX.EnableWindow(m_bConnectStatus);
    m_EditY.EnableWindow(m_bConnectStatus);
    m_EditZ.EnableWindow(m_bConnectStatus);
    m_EditR.EnableWindow(m_bConnectStatus);

    m_ButtonSendPTP.EnableWindow(m_bConnectStatus);
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CDobotDemoDlg::OnPaint()
{
    if (IsIconic())
    {
        CPaintDC dc(this); // device context for painting

        SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

        // Center icon in client rectangle
        int cxIcon = GetSystemMetrics(SM_CXICON);
        int cyIcon = GetSystemMetrics(SM_CYICON);
        CRect rect;
        GetClientRect(&rect);
        int x = (rect.Width() - cxIcon + 1) / 2;
        int y = (rect.Height() - cyIcon + 1) / 2;

        // Draw the icon
        dc.DrawIcon(x, y, m_hIcon);
    }
    else
    {
        CDialog::OnPaint();
    }
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CDobotDemoDlg::OnQueryDragIcon()
{
    return static_cast<HCURSOR>(m_hIcon);
}

void CDobotDemoDlg::OnBnClickedButtonConnect()//コネクトが押されたとき
{
    // TODO: Add your control notification handler code here
    if (!m_bConnectStatus) {
        if (ConnectDobot(0, 115200) != DobotConnect_NoError) {
            ::AfxMessageBox("Cannot connect Dobot!");
            return;
        }
        m_bConnectStatus = true;
        RefreshButtons();
        SetTimer(1, 250, NULL);//250ms
        InitDobot();
    } else {
        m_bConnectStatus = false;
        RefreshButtons();
        KillTimer(1);
        DisconnectDobot();
    }
}

void CDobotDemoDlg::InitDobot(void)//初期化
{
    // Command timeout
    SetCmdTimeout(3000);
    // Clear old commands and set the queued command running
    SetQueuedCmdClear();
    SetQueuedCmdStartExec();

    // Device SN
    char deviceSN[64];
    GetDeviceSN(deviceSN, sizeof(deviceSN));
    _cprintf("Device SN:%s\r\n", deviceSN);

    // Device Name
    char deviceName[64];
    GetDeviceName(deviceName, sizeof(deviceName));
    _cprintf("Device Name:%s\r\n", deviceName);

    // Device version information
    uint8_t majorVersion, minorVersion, revision;
    GetDeviceVersion(&majorVersion, &minorVersion, &revision);
    _cprintf("Device information:V%d.%d.%d\r\n", majorVersion, minorVersion, revision);

    // Set the end effector parameters
    EndEffectorParams endEffectorParams;
    memset(&endEffectorParams, 0, sizeof(EndEffectorParams));
    endEffectorParams.xBias = 71.6f;
    SetEndEffectorParams(&endEffectorParams, false, NULL);

    // 1. Set the JOG parameters
    JOGJointParams jogJointParams;
    for (uint32_t i = 0; i < 4; i++) {
        jogJointParams.velocity[i] = 200;
        jogJointParams.acceleration[i] = 200;
    }
    SetJOGJointParams(&jogJointParams, false, NULL);

    JOGCoordinateParams jogCoordinateParams;
    for (uint32_t i = 0; i < 4; i++) {
        jogCoordinateParams.velocity[i] = 200;
        jogCoordinateParams.acceleration[i] = 200;
    }
    SetJOGCoordinateParams(&jogCoordinateParams, false, NULL);

    JOGCommonParams jogCommonParams;
    jogCommonParams.velocityRatio = 50;
    jogCommonParams.accelerationRatio = 50;
    SetJOGCommonParams(&jogCommonParams, false, NULL);

    // 2. Set the PTP parameters
    PTPJointParams ptpJointParams;
    for (uint32_t i = 0; i < 4; i++) {
        ptpJointParams.velocity[i] = 200;
        ptpJointParams.acceleration[i] = 200;
    }
    SetPTPJointParams(&ptpJointParams, false, NULL);

    PTPCoordinateParams ptpCoordinateParams;
    ptpCoordinateParams.xyzVelocity = 200;
    ptpCoordinateParams.xyzAcceleration = 200;
    ptpCoordinateParams.rVelocity = 200;
    ptpCoordinateParams.rAcceleration = 200;
    SetPTPCoordinateParams(&ptpCoordinateParams, false, NULL);

    PTPJumpParams ptpJumpParams;
    ptpJumpParams.jumpHeight = 10;
    ptpJumpParams.zLimit = 150;
    SetPTPJumpParams(&ptpJumpParams, false, NULL);
}

BOOL CDobotDemoDlg::PreTranslateMessage(MSG *pMsg)
{
    HWND hWnd[] = {
        m_ButtonJ1P.m_hWnd, m_ButtonJ1N.m_hWnd,
        m_ButtonJ2P.m_hWnd, m_ButtonJ2N.m_hWnd,
        m_ButtonJ3P.m_hWnd, m_ButtonJ3N.m_hWnd,
        m_ButtonJ4P.m_hWnd, m_ButtonJ4N.m_hWnd
    };

    if (pMsg->message == WM_LBUTTONDOWN) {
        for (int i = 0; i < sizeof(hWnd) / sizeof(hWnd[0]); i++) {
            if (pMsg->hwnd == hWnd[i]) {
                JOGCmd jogCmd;
                jogCmd.isJoint = m_JOGMode.GetCurSel() == 0;
                jogCmd.cmd = i + 1;
                SetJOGCmd(&jogCmd, false, NULL);
                break;
            }
        }
    } else if (pMsg->message == WM_LBUTTONUP) {
        for (int i = 0; i < sizeof(hWnd) / sizeof(hWnd[0]); i++) {
            if (pMsg->hwnd == hWnd[i]) {
                JOGCmd jogCmd;
                jogCmd.isJoint = m_JOGMode.GetCurSel() == 0;
                jogCmd.cmd = 0;
                SetJOGCmd(&jogCmd, false, NULL);
                break;
            }
        }
    }
    return CDialog::PreTranslateMessage(pMsg);
}
//タイマールーチンであり、指定された時間毎にコールされる。（飛んでくる）
void CDobotDemoDlg::OnTimer(UINT_PTR nIDEvent)
{
    // TODO: Add your message handler code here and/or call default
    switch (nIDEvent) {
        case 1:
            do {
                Pose pose;
                if (GetPose(&pose) != DobotCommunicate_NoError) {
                    break;
                }
                CString str;
                str.Format("%1.3f", pose.jointAngle[0]);//角度
                m_StaticJ1.SetWindowText(str);
                str.Format("%1.3f", pose.jointAngle[1]);
                m_StaticJ2.SetWindowText(str);
                str.Format("%1.3f", pose.jointAngle[2]);
                m_StaticJ3.SetWindowText(str);
                str.Format("%1.3f", pose.jointAngle[3]);
                m_StaticJ4.SetWindowText(str);

                str.Format("%1.3f", pose.x);//座標
                m_StaticX.SetWindowText(str);
                str.Format("%1.3f", pose.y);
                m_StaticY.SetWindowText(str);
                str.Format("%1.3f", pose.z);
                m_StaticZ.SetWindowText(str);
                str.Format("%1.3f", pose.r);
                m_StaticR.SetWindowText(str);
            } while (0);
        break;

        default:

        break;
    }
    CDialog::OnTimer(nIDEvent);
}

void CDobotDemoDlg::GotoPoint(UINT mode, float x, float y, float z, float r, bool waitEnd)
{
    PTPCmd ptpCmd;

    ptpCmd.ptpMode = mode;
    ptpCmd.x = x;
    ptpCmd.y = y;
    ptpCmd.z = z;
    ptpCmd.r = r;

    // Send the command. If failed, just resend the command
    uint64_t queuedCmdIndex;
    do {
        int result = SetPTPCmd(&ptpCmd, true, &queuedCmdIndex);
        if (result == DobotCommunicate_NoError) {
            break;
        }
    } while (1);

     //Check whether the command is finished
    do {
        if (waitEnd == false) {
            break;
        }
        uint64_t currentIndex;
        int result = GetQueuedCmdCurrentIndex(&currentIndex);
        if (result == DobotCommunicate_NoError &&
            currentIndex >= queuedCmdIndex) {
                break;
        }
    } while (1);
}

void CDobotDemoDlg::GotoPointCP(UINT mode, float x, float y, float z, bool waitEnd)
{
	CPCmd cpCmd;

	cpCmd.cpMode = mode;
	cpCmd.x = x;
	cpCmd.y = y;
	cpCmd.z = z;
	//cpCmd.r = r;

	// Send the command. If failed, just resend the command
	uint64_t queuedCmdIndex;
	do {
		int result = SetCPCmd(&cpCmd, true, &queuedCmdIndex);
		if (result == DobotCommunicate_NoError) {
			break;
		}
	} while (1);

	//Check whether the command is finished
	do {
		if (waitEnd == false) {
			break;
		}
		uint64_t currentIndex;
		int result = GetQueuedCmdCurrentIndex(&currentIndex);
		if (result == DobotCommunicate_NoError &&
			currentIndex >= queuedCmdIndex) {
			break;
		}
	} while (1);
}

void CDobotDemoDlg::LaserCtrl(bool isOn, bool waitEnd)
{
    // Send the command. If failed, just resend the command
    uint64_t queuedCmdIndex;
    do {
        int result = SetEndEffectorLaser(true, isOn, true, &queuedCmdIndex);
        if (result == DobotCommunicate_NoError) {
            break;
        }
    } while (1);

    // Check whether the command is finished
    do {
        if (waitEnd == false) {
            break;
        }
        uint64_t currentIndex;
        int result = GetQueuedCmdCurrentIndex(&currentIndex);
        if (result == DobotCommunicate_NoError &&
            currentIndex >= queuedCmdIndex) {
                break;
        }
    } while (1);
}

void CDobotDemoDlg::SuctionCupCtrl(bool suck, bool waitEnd)
{
    // Send the command. If failed, just resend the command
    uint64_t queuedCmdIndex;
    do {
        int result = SetEndEffectorSuctionCup(true, suck, true, &queuedCmdIndex);
        if (result == DobotCommunicate_NoError) {
            break;
        }
    } while (1);

    // Check whether the command is finished
    do {
        if (waitEnd == false) {
            break;
        }
        uint64_t currentIndex;
        int result = GetQueuedCmdCurrentIndex(&currentIndex);
        if (result == DobotCommunicate_NoError &&
            currentIndex >= queuedCmdIndex) {
                break;
        }
    } while (1);
}

void CDobotDemoDlg::WaitForSeconds(float seconds, bool waitEnd)
{
    // Send the command. If failed, just resend the command
    uint64_t queuedCmdIndex;
    do {
        WAITCmd waitCmd;
        waitCmd.timeout = (uint32_t)(seconds * 1000);
        int result = SetWAITCmd(&waitCmd, true, &queuedCmdIndex);
        if (result == DobotCommunicate_NoError) {
            break;
        }
    } while (1);

    // Check whether the command is finished
    do {
        if (waitEnd == false) {
            break;
        }
        uint64_t currentIndex;
        int result = GetQueuedCmdCurrentIndex(&currentIndex);
        if (result == DobotCommunicate_NoError &&
            currentIndex >= queuedCmdIndex) {
                break;
        }
    } while (1);
}

void CDobotDemoDlg::Home(void)
{
    // Send the command. If failed, just resend the command
    uint64_t queuedCmdIndex;
    do {
        HOMECmd homeCmd;
        int result = SetHOMECmd(&homeCmd, true, &queuedCmdIndex);
        if (result == DobotCommunicate_NoError) {
            break;
        }
    } while (1);

    // Check whether the command is finished
    do {
        bool waitEnd = true;
        if (waitEnd == false) {
            break;
        }
        uint64_t currentIndex;
        int result = GetQueuedCmdCurrentIndex(&currentIndex);
        if (result == DobotCommunicate_NoError &&
            currentIndex >= queuedCmdIndex) {
                break;
        }
    } while (1);
}

void CDobotDemoDlg::OnBnClickedButtonSendPTP()
{
    // TODO: Add your control notification handler code here
    wchar_t buffer[32];

    m_EditX.GetWindowText((LPTSTR)buffer, sizeof(buffer));
    float x = (float)_wtof(buffer);
    m_EditY.GetWindowText((LPTSTR)buffer, sizeof(buffer));
    float y = (float)_wtof(buffer);
    m_EditZ.GetWindowText((LPTSTR)buffer, sizeof(buffer));
    float z = (float)_wtof(buffer);
    m_EditR.GetWindowText((LPTSTR)buffer, sizeof(buffer));
    float r = (float)_wtof(buffer);

    /*
     * In this demo, we just input the values and send the PTP command
     * In the real condition, there are two kinds of controlling of queued command
     * The first one, send the queued commands, and wait for command to be finished for every command
     * The second one, send the queued commands, and just wait for the last command to be finished!
     * The second one is much more effiency
     */

    // Just an example
    GotoPoint(PTPJUMPXYZMode, x, y, z, r, true);
}


void CDobotDemoDlg::OnBnClickedButton1()//+Xボタンが押されたとき
{
	float x, y, z, r;
	Pose pose;

	if (GetPose(&pose) != DobotCommunicate_NoError) 
	{
		;
	}
	else
	{
		x = pose.x + 10;
		y = pose.y;
		z = pose.z;
		r = pose.r;
		GotoPoint(PTPMOVLXYZMode, x, y, z, r, true);
	}
	// TODO: Add your control notification handler code here
}


void CDobotDemoDlg::OnBnClickedButton2()//-Xボタンが押されたとき
{
	float x, y, z, r;
	Pose pose;

	if (GetPose(&pose) != DobotCommunicate_NoError)
	{
		;
	}
	else
	{
		x = pose.x - 10;
		y = pose.y;
		z = pose.z;
		r = pose.r;
		GotoPoint(PTPMOVLXYZMode, x, y, z, r, true);
	}
	// TODO: Add your control notification handler code here
}


void CDobotDemoDlg::OnBnClickedButton3()//+Yボタンが押されたとき
{
	float x, y, z, r;
	Pose pose;

	if (GetPose(&pose) != DobotCommunicate_NoError)
	{
		;
	}
	else
	{
		x = pose.x;
		y = pose.y + 10;
		z = pose.z;
		r = pose.r;
		GotoPoint(PTPMOVLXYZMode, x, y, z, r, true);
	}
	// TODO: Add your control notification handler code here
}


void CDobotDemoDlg::OnBnClickedButton4()//-Yボタンが押されたとき
{
	float x, y, z, r;
	Pose pose;

	if (GetPose(&pose) != DobotCommunicate_NoError)
	{
		;
	}
	else
	{
		x = pose.x;
		y = pose.y - 10;
		z = pose.z;
		r = pose.r;
		GotoPoint(PTPMOVLXYZMode, x, y, z, r, true);
	}
	// TODO: Add your control notification handler code here
}


void CDobotDemoDlg::OnBnClickedButton5()//+Zボタンが押されたとき
{
	float x, y, z, r;
	Pose pose;

	if (GetPose(&pose) != DobotCommunicate_NoError)
	{
		;
	}
	else
	{
		x = pose.x;
		y = pose.y;
		z = pose.z + 10;
		r = pose.r;
		GotoPoint(PTPMOVLXYZMode, x, y, z, r, true);
	}
	// TODO: Add your control notification handler code here
}


void CDobotDemoDlg::OnBnClickedButton6()//-Zボタンが押されたとき
{
	float x, y, z, r;
	Pose pose;

	if (GetPose(&pose) != DobotCommunicate_NoError)
	{
		;
	}
	else
	{
		x = pose.x;
		y = pose.y;
		z = pose.z - 10;
		r = pose.r;
		GotoPoint(PTPMOVLXYZMode, x, y, z, r, true);
	}
	// TODO: Add your control notification handler code here
}


void CDobotDemoDlg::OnBnClickedButton7()//+Rボタンが押されたとき
{
	float x, y, z, r;
	Pose pose;

	if (GetPose(&pose) != DobotCommunicate_NoError)
	{
		;
	}
	else
	{
		x = pose.x;
		y = pose.y;
		z = pose.z;
		r = pose.r + 10;
		GotoPoint(PTPMOVLXYZMode, x, y, z, r, true);
	}
	// TODO: Add your control notification handler code here
}


void CDobotDemoDlg::OnBnClickedButton8()//-Rボタンが押されたとき
{
	float x, y, z, r;
	Pose pose;

	if (GetPose(&pose) != DobotCommunicate_NoError)
	{
		;
	}
	else
	{
		x = pose.x;
		y = pose.y;
		z = pose.z;
		r = pose.r - 10;
		GotoPoint(PTPMOVLXYZMode, x, y, z, r, true);
	}
	// TODO: Add your control notification handler code here
}


void CDobotDemoDlg::OnBnClickedButton10()
{
   //CFileDialog myDLG(TRUE, "cls", "*.cls", OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, "変換前のNCﾌｧｲﾙ(*.nc)");
	CFileDialog myDLG(TRUE);
	myDLG.m_ofn.lpstrInitialDir = _T("d:\\data\\H29_Suzuki");
	if (myDLG.DoModal() == IDOK)
	{
		CStdioFile fout(myDLG.GetPathName(), CFile::modeRead | CFile::typeText);
		m_edit.SetWindowText(myDLG.GetPathName());
	}
}

void CDobotDemoDlg::OnBnClickedButton12()
{
	HOMECmd gHOMECmd; //始動する前に原点（スタート）の位置を探す。
	uint64_t gQueuedCmdIndex;
	gQueuedCmdIndex = 1;
	SetHOMECmd(&gHOMECmd, 1, &gQueuedCmdIndex);
}



//////////////////////////////////////////////////////////////////////
void CDobotDemoDlg::OnBnClickedButton11()//Executeボタンが押されたとき
//////////////////////////////////////////////////////////////////////
{
	float x, y, z, r;
	float xw, yw, zw ; //ワーク座標

	char x_[30];
	char y_[30];
	char z_[30];

	int i, j;

	Pose pose;

	UpdateData(TRUE);//入力している値が、m_  にコピーされる。

	//HOMECmd gHOMECmd; //始動する前に原点（スタート）の位置を探す。
 //   uint64_t gQueuedCmdIndex;
	//gQueuedCmdIndex = 1;
	//SetHOMECmd(&gHOMECmd, 1, &gQueuedCmdIndex);

	if (GetPose(&pose) != DobotCommunicate_NoError)
	{
		;
	}
	else
	{
        xw = pose.x;
		yw = pose.y;
		zw = pose.z;
		r = pose.r;

//		GotoPoint(PTPMOVLXYZMode, xw, yw, zw, r, true); 

	}
	// TODO: Add your control notification handler code here

	FILE	*cls_file;//
	CString	name_cls; /* 変換前のCLSファイル名 */
	char buffer[100]; //CLSファイル内のある１行（文字）を格納するバッファ
	//namecls = name_cls;
	m_edit.GetWindowText(name_cls);
	if ((name_cls.Find("cls") == -1) && (name_cls.Find("CLS") == -1)) //CLSﾌｧｲﾙが選択されていなければ、エラー処理を行う。
	{
		MessageBox("変換前のCLSファイルが選択されていません", NULL, MB_OK | MB_ICONEXCLAMATION);
		return;
	}

	cls_file = fopen(name_cls, "rb");//cls_fileにはファイルを操作するハンドルが返ってくる
	if (cls_file == NULL)
	{
		MessageBox("変換前のCLSファイルをオープンできません。", NULL, MB_OK | MB_ICONEXCLAMATION);
		//m_check1.SetCheck(0);
		return;
	}

	while (1)
	{
		if (fgets(buffer, 200, cls_file) == NULL) //
			break;
		m_edit2.SetWindowText(buffer);

		i = 5;//冒頭の「GOTO/」の部分を省略したところから始める。
		j = 0;

		while (1) /*x座標のデータ*/ //buffer[]内の１行分データから、xの値を抽出する。
		{
			x_[j] = buffer[i];
			i++;
			j++;
			if (buffer[i] == ',')
			{
				x_[j] = NULL;//NULL=0x00;
				break;
			}
		}
		x = atof(x_);
		i++;
		j = 0;
		m_edit3.SetWindowText(x_);

		while (1) /*y座標のデータ*/ //buffer[]内の１行分データから、yの値を抽出する。
	   {
		    y_[j] = buffer[i];
		    i++;
		    j++;
		    if (buffer[i] == ',')
		  {
			y_[j] = NULL;//NULL=0x00;
			break;
		  }
	   }
		    y = atof(y_);
		    i++;
		    j = 0;
		    m_edit4.SetWindowText(y_);

	   while (1) /*z座標のデータ*/ //buffer[]内の１行分データから、zの値を抽出する。
	   {
		    z_[j] = buffer[i];
		    i++;
		    j++;
		    if (buffer[i] == ',')
		  {
			z_[j] = NULL;//NULL=0x00;
			break;
		  }
	   }
		    z = atof(z_);
		    i++;
		    j = 0;
		    m_edit5.SetWindowText(z_);

		    GotoPoint(PTPMOVLXYZMode, x+xw, y+yw, z+zw, r, false); //TODO: Add your control notification handler code here
	}

        fclose(cls_file);
}




void CDobotDemoDlg::OnBnClickedButton13()
{
	int sss;
	CString str;
//	m_velocity = m_velocity;
	UpdateData(TRUE);
//	m_velocity = m_velocity;
	// TODO: ここにコントロール通知ハンドラー コードを追加します。
	PTPCoordinateParams ptpCoordinateParams;//書き込み用
	PTPCoordinateParams ptpCoordinateParams2;//読み込み用

	ptpCoordinateParams.xyzVelocity = m_velocity;
	ptpCoordinateParams.xyzAcceleration = m_acceleration;
	ptpCoordinateParams.rVelocity = 200;//単位ms(Dobot Communication Protocol P.30より)
	ptpCoordinateParams.rAcceleration = 200;//単位ms
	sss=SetPTPCoordinateParams(&ptpCoordinateParams, false, NULL);
	GetPTPCoordinateParams(&ptpCoordinateParams2);
//	m_velocity2 = ptpCoordinateParams2.xyzVelocity;
//	m_acceleration2 = ptpCoordinateParams2.xyzAcceleration;
	str.Format("%1.f", ptpCoordinateParams2.xyzVelocity);//
	m_velocity3.SetWindowText(str); 
	str.Format("%1.f", ptpCoordinateParams2.xyzAcceleration);//
	m_acceleration3.SetWindowText(str);
	UpdateData(TRUE);
}

void CDobotDemoDlg::OnBnClickedButton14()//CP Executeボタンが押されたとき
{


	float x, y, z, r;
	float xw, yw, zw; //ワーク座標

	char x_[30];
	char y_[30];
	char z_[30];

	int i, j;

	Pose pose;

	CPCmd cpCmd;
//  cpCmd.cpMode = CPRelativeMode;//相対座標系
	cpCmd.cpMode = CPAbsoluteMode;//絶対座標系
	cpCmd.velocity = 100;

	UpdateData(TRUE);//入力している値が、m_  にコピーされる。

					 //HOMECmd gHOMECmd; //始動する前に原点（スタート）の位置を探す。
					 //   uint64_t gQueuedCmdIndex;
					 //gQueuedCmdIndex = 1;
					 //SetHOMECmd(&gHOMECmd, 1, &gQueuedCmdIndex);

	if (GetPose(&pose) != DobotCommunicate_NoError)
	{
		;
	}
	else
	{
		xw = pose.x;
		yw = pose.y;
		zw = pose.z;
		r = pose.r;

	 // GotoPoint(PTPMOVLXYZMode, xw, yw, zw, r, true); 

	}
	// TODO: Add your control notification handler code here

	FILE	*cls_file;//
	CString	name_cls; /* 変換前のCLSファイル名 */
	char buffer[100]; //CLSファイル内のある１行（文字）を格納するバッファ
					  //namecls = name_cls;
	m_edit.GetWindowText(name_cls);
	if ((name_cls.Find("cls") == -1) && (name_cls.Find("CLS") == -1)) //CLSﾌｧｲﾙが選択されていなければ、エラー処理を行う。
	{
		MessageBox("変換前のCLSファイルが選択されていません", NULL, MB_OK | MB_ICONEXCLAMATION);
		return;
	}

	cls_file = fopen(name_cls, "rb");//cls_fileにはファイルを操作するハンドルが返ってくる
	if (cls_file == NULL)
	{
		MessageBox("変換前のCLSファイルをオープンできません。", NULL, MB_OK | MB_ICONEXCLAMATION);
		//m_check1.SetCheck(0);
		return;
	}

	while (1)
	{
		if (fgets(buffer, 200, cls_file) == NULL) //
			break;
		m_edit2.SetWindowText(buffer);

		i = 5;//冒頭の「GOTO/」の部分を省略したところから始める。
		j = 0;

		while (1) /*x座標のデータ*/ //buffer[]内の１行分データから、xの値を抽出する。
		{
			x_[j] = buffer[i];
			i++;
			j++;
			if (buffer[i] == ',')
			{
				x_[j] = NULL;//NULL=0x00;
				break;
			}
		}
		x = atof(x_);
		i++;
		j = 0;
		m_edit3.SetWindowText(x_);

		while (1) /*y座標のデータ*/ //buffer[]内の１行分データから、yの値を抽出する。
		{
			y_[j] = buffer[i];
			i++;
			j++;
			if (buffer[i] == ',')
			{
				y_[j] = NULL;//NULL=0x00;
				break;
			}
		}
		y = atof(y_);
		i++;
		j = 0;
		m_edit4.SetWindowText(y_);

		while (1) /*z座標のデータ*/ //buffer[]内の１行分データから、zの値を抽出する。
		{
			z_[j] = buffer[i];
			i++;
			j++;
			if (buffer[i] == ',')
			{
				z_[j] = NULL;//NULL=0x00;
				break;
			}
		}
		z = atof(z_);
		i++;
		j = 0;
		m_edit5.SetWindowText(z_);

		//GotoPoint(PTPMOVLXYZMode, x + xw, y + yw, z + zw, r, true); //TODO: Add your control notification handler code here

	//cpCmd.x = x+xw;
	//cpCmd.y = y+yw;
	//cpCmd.z = z+zw;
	//SetCPCmd(&cpCmd, true, NULL);//CPモードでの絶対座標系の指令
	//GotoPointCP(CPAbsoluteMode, x + xw, y + yw, z + zw, false);

	}

	fclose(cls_file);


}

void CDobotDemoDlg::fwrite_pose(FILE* fp)
{
	Pose pose;
	static char crlf[3];
	crlf[0] = 0x0d;
	crlf[1] = 0x0a;
	crlf[2] = 0x00;
	static char kanma[3];
	kanma[0] = ',';
	kanma[1] = 0x00;

	GetPose(&pose);
	fprintf(fp, "%-.3f", pose.x);
	fputs(kanma, fp);
	fprintf(fp, "%-.3f", pose.y);
	fputs(kanma, fp);
	fprintf(fp, "%-.3f", pose.z);
	fputs(kanma, fp);
	fprintf(fp, "%-.3f", pose.r);
	fputs(kanma, fp);
	fprintf(fp, "%-.3f", pose.jointAngle[0]);
	fputs(kanma, fp);
	fprintf(fp, "%-.3f", pose.jointAngle[1]);
	fputs(kanma, fp);
	fprintf(fp, "%-.3f", pose.jointAngle[2]);
	fputs(kanma, fp);
	fprintf(fp, "%-.3f", pose.jointAngle[3]);
	fputs(crlf, fp);
}

void CDobotDemoDlg::OnBnClickedButton15()
{
	float x, y, z, r;
	Pose pose;
	int i;
	int j;
	int NUM = 10;
	FILE* fp;
	CString	name_fp;

	m_edit10.GetWindowText(name_fp);
	fp = fopen(name_fp, "wb");
	if (fp == NULL)
	{
		MessageBox("ファイルをオープンできません。", NULL, MB_OK | MB_ICONEXCLAMATION);
		return;
	}
	fwrite_pose(fp);

	
	for (i = 0; i < NUM; i++)
	{
		if (i % 2 == 0) {
			for (j = 0; j < NUM; j++)
			{
				if (GetPose(&pose) != DobotCommunicate_NoError)
				{
					break;
				}
				else
				{
					x = pose.x + 1;
					y = pose.y;
					z = pose.z;
					r = pose.r;
					GotoPoint(PTPMOVLXYZMode, x, y, z, r, true);
					//GetPose(&pose);
					fwrite_pose(fp);

				}
			}
		}
		else
		{
			for (j = 0; j < NUM; j++)
			{
				if (GetPose(&pose) != DobotCommunicate_NoError)
				{
					break;
				}
				else
				{
					x = pose.x - 1;
					y = pose.y;
					z = pose.z;
					r = pose.r;
					GotoPoint(PTPMOVLXYZMode, x, y, z, r, true);
					//GetPose(&pose);
					fwrite_pose(fp);

				}
			}
		}
		if (GetPose(&pose) != DobotCommunicate_NoError)
		{
			break;
		}
		else
		{
			x = pose.x;
			y = pose.y + 1;
			z = pose.z;
			r = pose.r;
			GotoPoint(PTPMOVLXYZMode, x, y, z, r, true);
			//GetPose(&pose);
			fwrite_pose(fp);

		}
	}
	fclose(fp);
}


void CDobotDemoDlg::OnBnClickedButton16()
{
	m_edit10.SetWindowText("c:\\data\\H31_Miki\\training.txt");
}
