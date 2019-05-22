// DobotDemoDlg.h : header file
//

#pragma once
#include "afxwin.h"


// CDobotDemoDlg dialog
class CDobotDemoDlg : public CDialog
{
// Construction
public:
    CDobotDemoDlg(CWnd* pParent = NULL);    // standard constructor

// Dialog Data
    enum { IDD = IDD_DOBOTDEMO_DIALOG };

    protected:
    virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
    virtual BOOL PreTranslateMessage(MSG *pMsg);
protected:
    HICON m_hIcon;

    // Generated message map functions
    virtual BOOL OnInitDialog();
    afx_msg void OnPaint();
    afx_msg HCURSOR OnQueryDragIcon();
    afx_msg void OnBnClickedButtonConnect();
    afx_msg void OnTimer(UINT_PTR nIDEvent);
    afx_msg void OnCbnSelchangeComboJOGMode();
    DECLARE_MESSAGE_MAP()
private:
    void InitControls(void);
    void InitDobot(void);
    void RefreshButtons(void);
private:
    // Wrap of Dobot API
    void GotoPoint(UINT mode, float x, float y, float z, float r, bool waitEnd = false);
	void GotoPointCP(UINT mode, float x, float y, float z, bool waitEnd = false);
    void LaserCtrl(bool isOn, bool waitEnd = false);
    void SuctionCupCtrl(bool suck, bool waitEnd = false);
    void WaitForSeconds(float seconds, bool waitEnd = false);
    void Home(void);
private:
    bool m_bConnectStatus;
    CButton m_ConnectButton;
    CComboBox m_JOGMode;

public:
    CButton m_ButtonJ1P;
    CButton m_ButtonJ1N;
    CButton m_ButtonJ2P;
    CButton m_ButtonJ2N;
    CButton m_ButtonJ3P;
    CButton m_ButtonJ3N;
    CButton m_ButtonJ4P;
    CButton m_ButtonJ4N;
    CStatic m_StaticJ1;
    CStatic m_StaticJ2;
    CStatic m_StaticJ3;
    CStatic m_StaticJ4;
    CStatic m_StaticX;
    CStatic m_StaticY;
    CStatic m_StaticZ;
    CStatic m_StaticR;
    CEdit m_EditX;
    CEdit m_EditY;
    CEdit m_EditZ;
    CEdit m_EditR;
    afx_msg void OnBnClickedButtonSendPTP();
    CButton m_ButtonSendPTP;
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton2();
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedButton4();
	afx_msg void OnBnClickedButton5();
	afx_msg void OnBnClickedButton6();
	afx_msg void OnBnClickedButton7();
	afx_msg void OnBnClickedButton8();
	afx_msg void OnBnClickedButton10();
	CEdit m_edit;
	afx_msg void OnBnClickedButton11();

	CEdit m_edit2;
	afx_msg void OnEnChangeEdit1();
	CEdit m_edit3;
	CEdit m_edit4;
	CEdit m_edit5;
	afx_msg void OnBnClickedButton12();
	int m_velocity;
	int m_acceleration;
	CEdit m_edit6;
	CEdit m_edit7;
	afx_msg void OnBnClickedButton13();
	//CString m_velocity2;
	//CString m_acceleration2;
	int m_velocity2;
	int m_acceleration2;
	CEdit m_velocity3;
	CEdit m_acceleration3;
	afx_msg void OnBnClickedButton14();
	afx_msg void OnBnClickedButton15();
	afx_msg void OnBnClickedButton16();
	CEdit m_edit10;
};
