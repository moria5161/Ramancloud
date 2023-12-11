#include <iostream>
#include <string>
#include <cmath>
#include "sg.h"

double gen_sg_coefficient_4_0(const double& m, const double& s)
{
	double molecular = (15*pow(m,4)+30*pow(m,3)-35*pow(m,2)-50*m+12)-35*(2*pow(m,2)+2*m-3)*pow(s,2)+63*pow(s,4);
	double denominator = (2*m+5)*(2*m+3)*(2*m+1)*(2*m-1)*(2*m-3);
	return (15*molecular)/(4*denominator);
}

double gen_sg_coefficient_2_0(const double& m,const double& s)
{
	double molecular = 3*(3*m*m+3*m-1-5*s*s);
	double denominator = (2*m+3)*(2*m+1)*(2*m-1);
	return molecular/denominator;
}

double gen_sg_coefficient(const int& n,const int& p,const int& m,const int& i)
{
	double gen_func;
	if(n == 2)
		gen_func = gen_sg_coefficient_2_0(m, i);
	if(n == 5 || n == 4)
		gen_func = gen_sg_coefficient_4_0(m, i);
	return gen_func;
}

int gen_sg_coefficients(const int& n,const int& p,const int& c, double weight[])
{
	int i,j;
	int m = (c - 1) / 2;
	j = 0;
	for(i = -1 * m; i < m + 1; i++)
		weight[j++] = gen_sg_coefficient(n,p,m,i);
	return j;
}

void move_average(double arry[], double weight[],const int& weight_sum, double smoothed_values[],const int& len)
{
	int i, j, k, m, n;
	int lx,ly;
	//double orign[len];
	double *orign = new double[len];
	double sum_y;
	k = weight_sum;

	m = ( k - 1 ) / 2;
	for(i = 0; i < len; i ++)
		orign[i] = arry[i];
	n = len ;
	for(i = 0; i < n; i ++)
	{
		lx = 0;
		ly = 0;
		if(i <= m-1)
		{
			lx = 0;
			ly = i + m;
		}
		else if( i >= n-m)
		{
			lx = i - m;
			ly = n;
		}
		else
		{
			lx = i - m;
			ly = i + m;
		}
		sum_y = 0;
		if(i <= m - 1)
			for(k = 0; k < m-i; k ++)
				sum_y += orign[0] * weight[k];
		else if(i >= n - m)
			for(k = 0; k < m-n+i+1; k ++)
				sum_y += orign[n-1] * weight[n - i + k + m];
		for(j = lx; j < ly; j++)
			sum_y += orign[j] * weight[j - i + m];
		smoothed_values[i] = sum_y;
	}
	delete[] orign;
}
void move_average_2(double arry[], double weight[], int weight_sum, double smoothed_values[], int len)
{
	int i, j, k, m, n;
	int lx,ly;
	//double orign[len];
	double *orign = new double[len];
	double sum_y;
	k = weight_sum;

	m = ( k - 1 ) / 2;
	for(i = 0; i < len; i ++)
		orign[i] = arry[i];
	n = len ;
	for(i = 0; i < n; i ++)
	{
		lx = 0;
		ly = 0;
		if(i <= m-1)
		{
			smoothed_values[i] = orign[i];
		}
		else if( i >= n-m)
		{
			smoothed_values[i] = orign[i];
		}
		else
		{
			sum_y = 0;
			for(j = i-m; j <= i+m; j++)
				sum_y += orign[j] * weight[j - i + m];
			smoothed_values[i] = sum_y;
		}
			
	}
	delete[] orign;
}
void basic_smooth(const int& n,const int& c, double arry[], double smoothed_values[],const int& len)
{
	int weight_sum;
	//double weight[len];
	double *weight = new double[len];
	weight_sum = gen_sg_coefficients(n, 0, c, weight);
	move_average(arry, weight, weight_sum, smoothed_values, len);
	delete[] weight;
}
void basic_smooth_2(const int& n, const int& c, double arry[], double smoothed_values[],const int& len)
{
	int weight_sum;
	//double weight[len];
	double *weight = new double[len];
	weight_sum = gen_sg_coefficients(n, 0, c, weight);
	move_average_2(arry, weight, weight_sum, smoothed_values, len);
	delete[] weight;
}
void noise_removal(double arry[], double smoothed_values[],const int& len)
{
	basic_smooth(5, 100, arry, smoothed_values, len);
}
