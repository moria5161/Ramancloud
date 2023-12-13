#include <iostream>
#include <cmath>
#include <string>
#include "sg.h"
#include "gaussion.h"
#include <vector>
using namespace std;

#define Ln 6
#define Lb 140

void get_mbk(double linear_data[],double cum[],const int& pl,const int& pr)
{
	int i;
	for(i = pl; i <= pr; i++)
		linear_data[i] = linear_data[i] + cum[i];
}

void cumsum(double arry[],const int& pl,const int& pr,double cum[])
{
	double temp = 0;
	int i;
	for(i = pl; i <= pr; i++) {
		temp += arry[i];
		cum[i] = temp;
	}
}

double Cal_mean(const int& pl,const int& pr,double arry[])
{
	int i;
	double temp = 0;
	for(i = pl; i <= pr; i++)
		temp += arry[i];
	return temp / (pr - pl + 1);
}

void diffline(double arry[],const int& pl,const int& pr,double diff[])
{
	double mean;
	int i;
	mean = Cal_mean(pl,pr,arry);
	for(i = pl; i <= pr; i++)
		diff[i] = arry[i] - mean;
}

double linear(const double& data,const double& a,const int& index)
{
	return a * index + data;
}

int adjust(int peaks[],int regions[][2],double background[], int peak_sum)
{
	int i,j;
	double a,pline;
	
	for(i = 0; i < peak_sum; i++) {
		if((regions[i][1] - regions[i][0]) < Ln) {
			if (i < peak_sum - 1) {
				for(j = i; j < peak_sum-1; j++){
					regions[j][0] = regions[j+1][0];
					regions[j][1] = regions[j+1][1];
					peaks[j] = peaks[j+1];
				}
			}
			--peak_sum;
		}
	}

	for(i = 0; i < peak_sum; i++) {

		double min = 0;
		int min_x = 0;

		a = (background[regions[i][1]]-background[regions[i][0]])/(regions[i][1] - regions[i][0]);
		for(j = regions[i][0]; j <= regions[i][1]; j++) {
			pline = linear(background[regions[i][0]],a,j-regions[i][0]);
			if((background[j] - pline) < min){
				min = background[j] - pline;
				min_x = j;
			}
			if(min < 0 && j == peaks[i]) {
				regions[i][0] = min_x;
				min = 0;
				min_x = 0;
			}
		}
		if(min < 0)
			regions[i][1] = min_x;
	}

	for(i = 0; i < peak_sum; i++) {
		if((regions[i][1] - regions[i][0]) < Ln){
			if (i < peak_sum - 1) {
				for(j = i; j < peak_sum-1; j++){
					regions[j][0] = regions[j+1][0];
					regions[j][1] = regions[j+1][1];
					peaks[j] = peaks[j+1];
				}
			}
			--peak_sum;
		}
	}
	return peak_sum;
}

void snds(double ds[], double ds_3[], const int& len)
{
	//double temp1[len];
	//double temp2[len];
	vector<double> temp1(len);
	vector<double> temp2(len);

	basic_smooth(2, Lb, ds, temp1.data(), len);
	basic_smooth(2, Lb, temp1.data(), temp2.data(), len);
	basic_smooth(2, Lb, temp2.data(), ds_3, len);
}

void derivative(double data[], double deriva[], const int& len)
{
	int i;
	for(i = 1; i < len; i++)
		deriva[i] = data[i] - data[i-1];
	deriva[0] = deriva[1];
}


void dg(double background[], double dgs[], const int& len)
{
	//double ds[len];
	//double ds_3[len];
	//double data[len];
	//double deriva[len];
	vector<double> ds(len);
	vector<double> ds_3(len);
	vector<double> data(len);
	vector<double> deriva(len);

	int i;
	basic_smooth(2, Ln, background, data.data(), len);
	derivative(data.data(), deriva.data(), len);
	basic_smooth(2, Ln, deriva.data(), ds.data(), len);
	snds(ds.data(), ds_3.data(), len);
	for(i = 0; i < len; i++)
		dgs[i] = ds[i] - ds_3[i];
}



int peak_region(double background[],int peaks[],int regions[][2],int is_smoothed[], const int& len)
{
	int i,j,p;
	//double dgs[len];
	vector<double> dgs(len);

	dg(background, dgs.data(), len);
	j = 0;

	for(i = 1; i < len; i++) {
		if(dgs[i-1] >= 0 && dgs[i] <= 0) {
			peaks[j] = i;
			if(fabs(dgs[i-1] - dgs[i]) < 0.01)
				is_smoothed[j] = 1;
			else
				is_smoothed[j] = 0;
			j++;
		}
	}

	int peak_sum = j;
	for(p = 0; p < peak_sum; p ++) {
		i = peaks[p] - 1;
		while(i > 0 && dgs[i - 1] >= 0)
			i -= 1;
		regions[p][0] = i;
		i = peaks[p] + 1;
		while( i < len - 1 && dgs[i + 1] <= 0)
			i += 1;
		if(i == len - 1)
			regions[p][1] = i;
		else
			regions[p][1] = i + 1;
	}

	return adjust(peaks, regions, background, peak_sum);
}

void get_background(int regions[][2], int is_smoothed[], double smoothed_values[], double arry[], const int& show,const int& len,const int& peak_sum)
{
	int i,j;
	double a,b;
	//double temp1[len],temp2[len];
	//double s3ds[len];
	//double diff[len];
	//double cum[len];
	vector<double> temp1(len), temp2(len);
	vector<double> s3ds(len);
	vector<double> diff(len);
	vector<double> cum(len);

	basic_smooth(2, Ln, smoothed_values, temp1.data(), len);
	derivative(temp1.data(), temp2.data(), len);
	basic_smooth(2, Ln, temp2.data(), temp1.data(), len);
	snds(temp1.data(), s3ds.data(), len);

	for(i = 0; i < peak_sum; i++) {
		if(is_smoothed[i] == 0) {
			b = smoothed_values[regions[i][0]];
			a = (smoothed_values[regions[i][1]] - smoothed_values[regions[i][0]])/(regions[i][1] - regions[i][0]);
			for(j = regions[i][0]; j <= regions[i][1]; j++)
				smoothed_values[j] = linear(b,a,j-regions[i][0]);
		}
	}

	if (show) {
		for(i = 0; i < peak_sum; i++) {
			diffline(s3ds.data(),regions[i][0],regions[i][1],diff.data());
			cumsum(diff.data(),regions[i][0],regions[i][1],cum.data());
			get_mbk(smoothed_values,cum.data(),regions[i][0],regions[i][1]);
		}
	}
}

void repeat(double smoothed_values[], double arry[], double background[], const int& len)
{
	int i,j;
	int sig;
	int peak_sum;
	//int peaks[len];
	//int regions[len][2];
	//int is_smoothed[len];
	vector<int> peaks(len);
	vector< int[2] > regions(len);
	vector<int> is_smoothed(len);

	for(i = 0; i < len; i++)
		background[i] = smoothed_values[i];
	i = 0;
	sig = 0;

	while(!sig) {
		peak_sum = peak_region(background,peaks.data(),regions.data(),is_smoothed.data(), len);
		i += 1;
		if(i > 5)
			sig = 1;
		else {
			int flag = 1;
			for(j = 0; j < peak_sum; j++)
				flag *= is_smoothed[j];
			if(flag)
				sig = 1;
		}
		get_background(regions.data(), is_smoothed.data(), background, arry, sig, len, peak_sum);
	}
}


void gaussion(double x[], double y[],const int& len)
{
	//double smoothed_values[len];
	//double background[len];
	//double background_smooth[len];
	vector<double> smoothed_values(len);
	vector<double> background(len);
	vector<double> background_smooth(len);

	noise_removal(y, smoothed_values.data(), len);
	repeat(smoothed_values.data(), x, background.data(), len);
	basic_smooth(5, Lb, background.data(), background_smooth.data(), len);

	int i;
	for (i = 0; i < len; i++)
		y[i] -= background_smooth[i];
}


