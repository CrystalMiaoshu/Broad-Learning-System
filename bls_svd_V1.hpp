#include"iostream"
#include"random"
#include"Eigen/Eigen" 
#include"ctime"
#include"fstream"
#include"stdio.h"
#include"string.h"
#include<Eigen/Dense>
#include<Eigen/SVD>
using namespace std;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

struct increment
{
	int position;
	Eigen::MatrixXd weight_from_input_to_mapNode_new;
	Eigen::MatrixXd belta_from_input_to_mapNode_new;
	Eigen::MatrixXd weight_from_mapNode_to_enhanceNode_new;
	Eigen::MatrixXd belta_from_mapNode_to_enhanceNode_new;
	Eigen::MatrixXd VP;
};

class bls_network
{
private:
	double threshold_for_testData;
	int time_seed = 0;
	int mapNode_number;
	int mapNode_size;
	int enhanceNode_number;
	int enhanceNode_size;
	int input_dimension;
	int output_dimension;
	int trainData_size;
	int testData_size;

	Eigen::MatrixXd train_data;
	Eigen::MatrixXd train_label;
	Eigen::MatrixXd test_data;
	Eigen::MatrixXd test_label;
	Eigen::MatrixXd weight_from_input_to_mapNode;
	Eigen::MatrixXd weight_from_mapNode_to_enhanceNode;
	Eigen::MatrixXd belta_from_input_to_mapNode;
	Eigen::MatrixXd belta_from_mapNode_to_enhanceNode;
	Eigen::MatrixXd belta_from_input_to_mapNode_row;
	Eigen::MatrixXd belta_from_mapNode_to_enhanceNode_row;
	Eigen::MatrixXd mapNode;
	Eigen::MatrixXd enhanceNode;
	Eigen::MatrixXd comprehensiveNode;
	Eigen::MatrixXd weight_from_comprehensiveNode_to_output;
	Eigen::MatrixXd pseudo_inverse_comprehensiveNode;
	Eigen::MatrixXd predict_label;
	vector<increment> increment_nodes;

public:
	bls_network(double threshold);
	void init_parameter();
	void load_data();
	Eigen::MatrixXd create_random_matrix(int rows, int cols);
	Eigen::MatrixXd sigmod(Eigen::MatrixXd LM);
	Eigen::MatrixXd relu(Eigen::MatrixXd LM);
	Eigen::MatrixXd vstack(Eigen::MatrixXd matrix_1, Eigen::MatrixXd matrix_2);
	Eigen::MatrixXd hstack(Eigen::MatrixXd matrix_1, Eigen::MatrixXd matrix_2);
	Eigen::MatrixXd calculate_pseudo_inverse(Eigen::MatrixXd matrix);
	vector<Eigen::MatrixXd> h_compose_matrix_to_vector(Eigen::MatrixXd matrix, int matrix_size);
	Eigen::MatrixXd h_stack_vector_to_matrix(vector<Eigen::MatrixXd> vector);
	vector<Eigen::MatrixXd> calculate_VP(vector<Eigen::MatrixXd> matrix_vector);
	void plus_VP(vector<Eigen::MatrixXd> &matrix_vector, vector<Eigen::MatrixXd> VP_vector);
	void svd(Eigen::MatrixXd matrix, Eigen::MatrixXd &VP_matrix);
	void forward();
	void increment_enhanceNode();
	void increment_mapNode();
	void increment_input(Eigen::MatrixXd new_input);
	void predict();
	double accuracy();
	double calculate_accuracy(Eigen::MatrixXd real, Eigen::MatrixXd predict);
};
