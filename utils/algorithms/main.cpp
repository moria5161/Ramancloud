#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "gaussion.h"
using namespace std;

// ��ȡ�����ļ�
void readSample(ifstream &ifs, vector<double> &x, vector<double> &y);
// �������ļ�
void saveResult(ofstream &ofs, const vector<double> &x, vector<double> &y);

int main(int argc, char *argv[])
{
	// argv[1] �������ļ�
	ifstream ifs;
	ifs.open(argv[1]);
	if (!ifs.is_open())
	{
		cerr << "�������ļ���" << argv[1] << "ʧ�ܣ�" << endl;
		exit(-1);
	}
	// argv[2] ������ļ�
	ofstream ofs;
	ofs.open(argv[2], ios::trunc);
	if (!ofs.is_open())
	{
		cerr << "������ļ���" << argv[2] << "ʧ�ܣ�" << endl;
		exit(-1);
	}
	// ����������ȡ����
	vector<double> x, y;
	readSample(ifs, x, y);
	// ����
	gaussion(x.data(), y.data(), x.size());
	// ����������
	saveResult(ofs, x, y);
	// �ر��ļ�
	ifs.close();
	ofs.close();

	return 0;
}

void readSample(ifstream &ifs, vector<double> &x, vector<double> &y)
{
	//// ����ǰ5��
	//string skipLine;
	//for (int i = 0; i < 5; i++)
	//{
	//	getline(ifs, skipLine);
	//}
	// ��ȡ
	double xi, yi;
	//char comma, separator;
	while (!ifs.eof())
	{
		//ifs >> xi >> comma >> yi >> separator;
		ifs >> xi >> yi;
		x.push_back(xi);
		y.push_back(yi);
	};
}

void saveResult(ofstream &ofs, const vector<double> &x, vector<double> &y)
{
	for (size_t i = 0; i < x.size(); i++)
	{
		ofs << x[i] << " " << y[i] << endl;
	}
}