#include<iostream>
#include<random>
#include<Eigen/Eigen>
#include<ctime>
#include"bls_svd_V1.hpp"

using namespace Eigen;
using namespace std;

int main()
{
	cout << "start" << endl;
	bls_network bls(0.9);
	bls.load_data();
	bls.forward();
	bls.increment_enhanceNode();
	for (int i = 0; i < 1; i++)
	{
		bls.increment_enhanceNode();
	}
	//bls.increment_mapNode();
	system("Pause");


	return 0;
}

/*int main()
{
	Eigen::MatrixXd A = Eigen::MatrixXd::Zero(0, 0);
	cout<<A.rows()<<"    "<<A.cols()<<endl;
}*/