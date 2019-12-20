#include"bls_svd_V1.hpp"

/*
**初始化构造函数
**threshold:测试验证集精度的阈值
*/
bls_network::bls_network(double threshold)
{
	threshold_for_testData = threshold;
}

vector<double> readFile(string file_name)// 文件读取函数
{
	fstream open_file;
	open_file.open(file_name, ios::in);
	vector<double> parameter;
	char buffer[256];
	char *after_compose_char;
	double num;

	while (!open_file.eof())//判断是否是文件末尾
	{
		const char *compose_mark = " ,";
		char *residual_str = NULL;
		open_file.getline(buffer, 256, '\n');//接收256个字符到buffer，最后一个为'\n'
		after_compose_char = strtok_s(buffer, compose_mark, &residual_str);//以compose_mark为分隔标志，存至最后一个形参
		while (after_compose_char)
		{
			//cout<<after_compose_char<<"   ";
			sscanf_s(after_compose_char, "%lf", &num, NULL);
			//cout<<num<<endl;
			parameter.push_back(num);
			after_compose_char = strtok_s(NULL, compose_mark, &residual_str);
		}
	}
	/*for(int i=0; i<parameter.size(); i++)
	{
		cout<<parameter[i]<<endl;
	}*/
	open_file.close();
	return parameter;
}

void bls_network::init_parameter()
{
	//vector<int> parameter_for_data;
	vector<double> parameter_for_dimension;
	vector<double> parameter_for_number_and_size;
	//parameter_for_data = readFile("../src/data_size.txt"); 
	parameter_for_dimension = readFile("C:/Users/shu.miao/Desktop/新建文件夹/src/dimension.txt");
	parameter_for_number_and_size = readFile("C:/Users/shu.miao/Desktop/新建文件夹/src/number_and_size.txt");
	//trainData_size = parameter_for_data[0];
	//testData_size = parameter_for_data[1];	
	mapNode_number = int(parameter_for_number_and_size[0]);
	mapNode_size = int(parameter_for_number_and_size[1]);
	enhanceNode_number = int(parameter_for_number_and_size[2]);
	enhanceNode_size = int(parameter_for_number_and_size[3]);
	input_dimension = int(parameter_for_dimension[0]);
	output_dimension = int(parameter_for_dimension[1]);
}

Eigen::MatrixXd assignment_for_matrix(vector<double> V, int V_dimension)//V就是整体1Xmn，V_dimension=n,把1*MxN=MXN
{
	Eigen::MatrixXd matrix;
	matrix = Eigen::MatrixXd::Zero(V.size() / V_dimension, V_dimension);//创建一个0矩阵
	for (int i = 0; i < V.size() / V_dimension; i++)
	{
		for (int j = 0; j < V_dimension; j++)
		{
			matrix(i, j) = V[j + i * V_dimension];
		}
	}
	return matrix;
}

void bls_network::load_data()
{
	bls_network::init_parameter();
	vector<double> train_init_data;
	vector<double> train_init_label;
	vector<double> test_init_data;
	vector<double> test_init_label;
	train_init_data = readFile("C:/Users/shu.miao/Desktop/新建文件夹/data/train_data.txt");
	train_init_label = readFile("C:/Users/shu.miao/Desktop/新建文件夹/data/train_label.txt");
	test_init_data = readFile("C:/Users/shu.miao/Desktop/新建文件夹/data/test_data.txt");
	test_init_label = readFile("C:/Users/shu.miao/Desktop/新建文件夹/data/test_label.txt");
	trainData_size = train_init_data.size() / input_dimension;
	testData_size = test_init_data.size() / input_dimension;
	train_data = assignment_for_matrix(train_init_data, input_dimension);
	train_label = assignment_for_matrix(train_init_label, output_dimension);
	test_data = assignment_for_matrix(test_init_data, input_dimension);
	test_label = assignment_for_matrix(test_init_label, output_dimension);
	cout<<train_data(0,0)<<endl;
}

/*
void bls_network::update_time_seed()
{
	;
}

void bls_network::assign_for_belta()
{
	for(int i=0; i<trainData_size; i++)
	{
		belta_for_enhanceNodes.row(i) = belta_for_enhanceNode_row;
		belta_for_mapNodes.row(i) = belta_for_mapNode_row;
	}
}

void bls_network::init_weight_and_belta()
{
	srand((unsigned)time(NULL)+time_seed);
	time_seed = time_seed+1000;
	weight_from_input_to_mapNodes = Eigen::MatrixXd::Random(input_dimension, mapNode_number*mapNode_size);
	srand((unsigned)time(NULL)+time_seed);
	time_seed = time_seed+1000;
	weight_from_mapNode_to_enhanceNode = Eigen::MatrixXd::Random(mapNode_number*mapNode_size, enhanceNode_number*enhanceNode_size);
	srand((unsigned)time(NULL)+time_seed);
	time_seed = time_seed+1000;
	belta_for_mapNode_row = Eigen::MatrixXd::Random(1, mapNode_number*mapNode_size);
	srand((unsigned)time(NULL)+time_seed);
	time_seed = time_seed+1000;
	belta_for_enhanceNode_row = Eigen::MatrixXd::Random(1, enhanceNode_number*enhanceNode_size);
	bls_network::assign_for_belta();
}*/

Eigen::MatrixXd bls_network::sigmod(Eigen::MatrixXd LM)//sigmode激活函数
{
	Eigen::MatrixXd LM_minus;
	Eigen::MatrixXd LM_minus_exp;
	Eigen::MatrixXd I;
	I = Eigen::MatrixXd::Ones(LM.rows(), LM.cols());
	LM_minus = -LM;
	LM_minus_exp = LM_minus.array().exp();
	LM_minus_exp = LM_minus_exp + I;
	return I.array() / LM_minus_exp.array();
}

Eigen::MatrixXd bls_network::relu(Eigen::MatrixXd LM)
{
	Eigen::MatrixXd abs_LM;
	abs_LM = LM.array().abs();
	abs_LM = abs_LM + LM * 0.5;
	return abs_LM / 2;
}

void assignment_for_belta(Eigen::MatrixXd ass_matrix, Eigen::MatrixXd &to_ass_matrix, int number_of_ass_row)// 把每一行的1XK付给MXK
{
	int cols = ass_matrix.cols();
	to_ass_matrix = Eigen::MatrixXd::Zero(number_of_ass_row, cols);
	for (int i = 0; i < number_of_ass_row; i++)
	{
		to_ass_matrix.row(i) = ass_matrix;
	}
}

Eigen::MatrixXd bls_network::create_random_matrix(int rows, int cols)
{
	Eigen::MatrixXd matrix;
	srand((unsigned)time(NULL) + time_seed);
	time_seed = time_seed + 1000;
	matrix = Eigen::MatrixXd::Random(rows, cols);
	return matrix;
}

Eigen::MatrixXd bls_network::hstack(Eigen::MatrixXd matrix_1, Eigen::MatrixXd matrix_2)
{
	int rows = matrix_1.rows();
	int cols_1 = matrix_1.cols();
	int cols_2 = matrix_2.cols();
	Eigen::MatrixXd matrix_plus;
	matrix_plus = Eigen::MatrixXd::Zero(rows, cols_1 + cols_2);
	matrix_plus.block(0, 0, rows, cols_1) = matrix_1;
	matrix_plus.block(0, cols_1, rows, cols_2) = matrix_2;
	return matrix_plus;
}

Eigen::MatrixXd bls_network::vstack(Eigen::MatrixXd matrix_1, Eigen::MatrixXd matrix_2)
{
	int rows_1 = matrix_1.rows();
	int rows_2 = matrix_2.rows();
	int cols = matrix_1.cols();
	Eigen::MatrixXd matrix_plus;
	matrix_plus = Eigen::MatrixXd::Zero(rows_1 + rows_2, cols);
	matrix_plus.block(0, 0, rows_1, cols) = matrix_1;
	matrix_plus.block(rows_1, 0, rows_2, cols) = matrix_2;
	return matrix_plus;
}

Eigen::MatrixXd bls_network::calculate_pseudo_inverse(Eigen::MatrixXd matrix)//将所有向量矩阵放到一个容器中
{
	Eigen::MatrixXd matrixT;
	Eigen::MatrixXd matrixT_plus_matrix;
	Eigen::MatrixXd matrixT_plus_matrix_I;
	matrixT = matrix.transpose();
	matrixT_plus_matrix = matrixT * matrix;
	//cout<<matrixT_plus_matrix.rows()<<"   "<<matrixT_plus_matrix.cols()<<endl;
	matrixT_plus_matrix_I = matrixT_plus_matrix.inverse();
	//cout<<(matrixT_plus_matrix_I*matrixT_plus_matrix).determinant()<<endl;
	//cout<<matrixT_plus_matrix_I*matrixT*matrix<<endl;
	return matrixT_plus_matrix_I * matrixT;
}

void bls_network::svd(Eigen::MatrixXd matrix, Eigen::MatrixXd &VP_matrix)//返回一个VP_matrix
{
	Eigen::MatrixXd matrix_A;
	Eigen::MatrixXd matrix_V;
	int P = 0;
	JacobiSVD<Eigen::MatrixXd> svd(matrix, ComputeFullU | ComputeFullV);
	matrix_V = svd.matrixV();
	matrix_A = svd.singularValues();
	for (int i = 0; i < matrix_A.rows(); i++)
	{
		if (matrix_A(i) > 0.1 && i < matrix_A.rows() - 1)
		{
			;
		}
		else if (matrix_A(i) > 0.1 && i == matrix_A.rows() - 1)
		{
			P = matrix_A.rows() - 1;
		}
		else
		{
			P = i;
			break;
		}
	}
	VP_matrix = matrix_V.block(0, 0, matrix_V.rows(), P);
}

vector<Eigen::MatrixXd> bls_network::h_compose_matrix_to_vector(Eigen::MatrixXd matrix, int matrix_size)//分出块的列数，matrix_size=k/mapnumber,marix=MXK
{
	vector<Eigen::MatrixXd> matrix_vector;
	int matrix_number = matrix.cols() / matrix_size;
	for (int i = 0; i < matrix_number; i++)
	{
		Eigen::MatrixXd matrix_part;
		matrix_part = matrix.block(0, i*matrix_size, matrix.rows(), matrix_size);
		matrix_vector.push_back(matrix_part);
	}
	return matrix_vector;
}

Eigen::MatrixXd bls_network::h_stack_vector_to_matrix(vector<Eigen::MatrixXd> vector)
{
	Eigen::MatrixXd matrix = vector[0];
	for (int i = 1; i < vector.size(); i++)
	{
		matrix = bls_network::hstack(matrix, vector[i]);
	}
	return matrix;
}

vector<Eigen::MatrixXd> bls_network::calculate_VP(vector<Eigen::MatrixXd> matrix_vector)//将所有向量矩阵放到一个容器中
{
	vector<Eigen::MatrixXd> VP_vector;
	for (int i = 0; i < matrix_vector.size(); i++)
	{
		Eigen::MatrixXd matrix = matrix_vector[i];
		Eigen::MatrixXd VP_matrix;
		bls_network::svd(matrix, VP_matrix);
		VP_vector.push_back(VP_matrix);
	}
	return VP_vector;
}

void bls_network::plus_VP(vector<Eigen::MatrixXd> &matrix_vector, vector<Eigen::MatrixXd> VP_vector)//每个子块都乘以VP
{
	for (int i = 0; i < matrix_vector.size(); i++)
	{
		Eigen::MatrixXd matrix;
		matrix = matrix_vector[i];
		matrix = matrix * VP_vector[i];
		matrix_vector[i] = matrix;
	}
}

void bls_network::forward()
{
	vector<Eigen::MatrixXd> mapNode_vector;
	vector<Eigen::MatrixXd> enhanceNode_vector;
	vector<Eigen::MatrixXd> weight_from_input_vector;
	vector<Eigen::MatrixXd> belta_from_input_vector;
	vector<Eigen::MatrixXd> weight_from_mapNode_vector;
	vector<Eigen::MatrixXd> belta_from_mapNode_vector;
	vector<Eigen::MatrixXd> VP_matrix_of_mapNode_vector;
	vector<Eigen::MatrixXd> VP_matrix_of_enhanceNode_vector;
	Eigen::MatrixXd VP_of_comprehensiveNode;
	weight_from_input_to_mapNode = bls_network::create_random_matrix(input_dimension, mapNode_size*mapNode_number);//创建一个随机矩阵NXK
	belta_from_input_to_mapNode_row = bls_network::create_random_matrix(1, mapNode_size*mapNode_number);//创建一个随机矩阵用于偏置
	assignment_for_belta(belta_from_input_to_mapNode_row, belta_from_input_to_mapNode, trainData_size);//为偏置赋值
	mapNode = train_data * weight_from_input_to_mapNode;
	mapNode = mapNode + belta_from_input_to_mapNode;
	//cout<<"mapNode_1:"<<mapNode.rows()<<"   "<<mapNode.cols()<<endl;
	mapNode_vector = bls_network::h_compose_matrix_to_vector(mapNode, mapNode_size);//三行全分块
	weight_from_input_vector = bls_network::h_compose_matrix_to_vector(weight_from_input_to_mapNode, mapNode_size);
	belta_from_input_vector = bls_network::h_compose_matrix_to_vector(belta_from_input_to_mapNode_row, mapNode_size);
	VP_matrix_of_mapNode_vector = bls_network::calculate_VP(mapNode_vector);//计算VP
	bls_network::plus_VP(mapNode_vector, VP_matrix_of_mapNode_vector);
	bls_network::plus_VP(weight_from_input_vector, VP_matrix_of_mapNode_vector);
	bls_network::plus_VP(belta_from_input_vector, VP_matrix_of_mapNode_vector);
	mapNode = bls_network::h_stack_vector_to_matrix(mapNode_vector);//
	mapNode = sigmod(mapNode);
	weight_from_input_to_mapNode = bls_network::h_stack_vector_to_matrix(weight_from_input_vector);
	belta_from_input_to_mapNode_row = bls_network::h_stack_vector_to_matrix(belta_from_input_vector);
	//cout<<belta_from_input_to_mapNode_row.cols()<<endl;
	//cout<<"mapNode_2:"<<mapNode.rows()<<"   "<<mapNode.cols()<<endl;
	//cout<<"weight:"<<weight_from_input_to_mapNode.rows()<<"   "<<weight_from_input_to_mapNode.cols()<<endl;
	weight_from_mapNode_to_enhanceNode = bls_network::create_random_matrix(mapNode.cols(), enhanceNode_size*enhanceNode_number);
	belta_from_mapNode_to_enhanceNode_row = bls_network::create_random_matrix(1, enhanceNode_size*enhanceNode_number);
	assignment_for_belta(belta_from_mapNode_to_enhanceNode_row, belta_from_mapNode_to_enhanceNode, trainData_size);
	enhanceNode = mapNode * weight_from_mapNode_to_enhanceNode;
	enhanceNode = enhanceNode + belta_from_mapNode_to_enhanceNode;
	weight_from_mapNode_vector = bls_network::h_compose_matrix_to_vector(weight_from_mapNode_to_enhanceNode, enhanceNode_size);
	belta_from_mapNode_vector = bls_network::h_compose_matrix_to_vector(belta_from_mapNode_to_enhanceNode_row, enhanceNode_size);
	enhanceNode_vector = bls_network::h_compose_matrix_to_vector(enhanceNode, enhanceNode_size);
	VP_matrix_of_enhanceNode_vector = bls_network::calculate_VP(enhanceNode_vector);
	bls_network::plus_VP(enhanceNode_vector, VP_matrix_of_enhanceNode_vector);
	bls_network::plus_VP(weight_from_mapNode_vector, VP_matrix_of_enhanceNode_vector);
	bls_network::plus_VP(belta_from_mapNode_vector, VP_matrix_of_enhanceNode_vector);
	enhanceNode = bls_network::h_stack_vector_to_matrix(enhanceNode_vector);
	enhanceNode = bls_network::relu(enhanceNode);
	weight_from_mapNode_to_enhanceNode = h_stack_vector_to_matrix(weight_from_mapNode_vector);
	belta_from_mapNode_to_enhanceNode_row = h_stack_vector_to_matrix(belta_from_mapNode_vector);
	//cout<<"P_mapNode: "<<P_of_mapNode<<"  P_enhanceNode:"<<P_of_enhanceNode<<endl;
	//cout<<"V_mapNode_row:  "<<V_of_mapNode.rows()<<"V_mapNode_col:"<<V_of_mapNode.cols()<<endl;
	//cout<<"V_enhanceNode_row:  "<<V_of_enhanceNode.rows()<<"V_enhanceNode_col:"<<V_of_enhanceNode.cols()<<endl;   
	comprehensiveNode = bls_network::hstack(mapNode, enhanceNode);
	bls_network::svd(comprehensiveNode, VP_of_comprehensiveNode);
	pseudo_inverse_comprehensiveNode = bls_network::calculate_pseudo_inverse(comprehensiveNode*VP_of_comprehensiveNode);
	comprehensiveNode = comprehensiveNode * VP_of_comprehensiveNode;
	//cout<<pseudo_inverse_comprehensiveNode.rows()<<"   "<<pseudo_inverse_comprehensiveNode.cols()<<endl;
	weight_from_comprehensiveNode_to_output = pseudo_inverse_comprehensiveNode * train_label;
	double a = bls_network::calculate_accuracy(train_label, comprehensiveNode*weight_from_comprehensiveNode_to_output);
	cout << a << endl;
	//cout<<weight_from_comprehensiveNode_to_output<<endl;
}


void bls_network::increment_enhanceNode()
{
	Eigen::MatrixXd VP_of_enhanceNode_new;
	Eigen::MatrixXd weight_from_mapNode_to_enhanceNode_N;
	Eigen::MatrixXd belta_from_mapNode_to_enhanceNode_row_N;
	Eigen::MatrixXd belta_from_mapNode_to_enhanceNode_N;
	Eigen::MatrixXd enhanceNode_new;
	Eigen::MatrixXd B_matrixT;
	Eigen::MatrixXd C_matrix;
	Eigen::MatrixXd D_matrix;
	Eigen::MatrixXd I_matrix;
	vector<Eigen::MatrixXd> weight_from_mapNode_vector;
	vector<Eigen::MatrixXd> belta_from_mapNode_vector;
	vector<Eigen::MatrixXd> enhanceNode_vector;
	vector<Eigen::MatrixXd> VP_matrix_of_enhanceNode_vector;

	increment increment_for_enhanceNode;
	weight_from_mapNode_to_enhanceNode_N = bls_network::create_random_matrix(mapNode.cols(), enhanceNode_size);
	belta_from_mapNode_to_enhanceNode_row_N = bls_network::create_random_matrix(1, enhanceNode_size);
	assignment_for_belta(belta_from_mapNode_to_enhanceNode_row_N, belta_from_mapNode_to_enhanceNode_N, trainData_size);
	enhanceNode_new = mapNode * weight_from_mapNode_to_enhanceNode_N;
	enhanceNode_new = enhanceNode_new + belta_from_mapNode_to_enhanceNode_N;
	enhanceNode_vector = bls_network::h_compose_matrix_to_vector(enhanceNode_new, enhanceNode_size);
	weight_from_mapNode_vector = bls_network::h_compose_matrix_to_vector(weight_from_mapNode_to_enhanceNode_N, enhanceNode_size);
	belta_from_mapNode_vector = bls_network::h_compose_matrix_to_vector(belta_from_mapNode_to_enhanceNode_row_N, enhanceNode_size);
	VP_matrix_of_enhanceNode_vector = bls_network::calculate_VP(enhanceNode_vector);
	bls_network::plus_VP(enhanceNode_vector, VP_matrix_of_enhanceNode_vector);
	bls_network::plus_VP(weight_from_mapNode_vector, VP_matrix_of_enhanceNode_vector);
	bls_network::plus_VP(belta_from_mapNode_vector, VP_matrix_of_enhanceNode_vector);
	enhanceNode_new = bls_network::h_stack_vector_to_matrix(enhanceNode_vector);
	weight_from_mapNode_to_enhanceNode_N = bls_network::h_stack_vector_to_matrix(weight_from_mapNode_vector);
	belta_from_mapNode_to_enhanceNode_row_N = bls_network::h_stack_vector_to_matrix(belta_from_mapNode_vector);
	enhanceNode_new = bls_network::relu(enhanceNode_new);
	D_matrix = pseudo_inverse_comprehensiveNode * enhanceNode_new;
	C_matrix = enhanceNode_new - comprehensiveNode * D_matrix;
	I_matrix = Eigen::MatrixXd::Ones(D_matrix.cols(), D_matrix.cols());
	if (C_matrix.all())
	{
		B_matrixT = I_matrix + D_matrix.transpose()*D_matrix;
		B_matrixT = B_matrixT.inverse();
		B_matrixT = B_matrixT * D_matrix.transpose()*pseudo_inverse_comprehensiveNode;
	}
	else
	{
		B_matrixT = calculate_pseudo_inverse(C_matrix);
	}
	comprehensiveNode = bls_network::hstack(comprehensiveNode, enhanceNode_new);
	pseudo_inverse_comprehensiveNode = pseudo_inverse_comprehensiveNode - D_matrix * B_matrixT;
	pseudo_inverse_comprehensiveNode = bls_network::vstack(pseudo_inverse_comprehensiveNode, B_matrixT);
	//cout<<D_matrix.rows()<<"   "<<D_matrix.cols()<<endl;
	//cout<<B_matrixT.rows()<<"    "<<B_matrixT.cols()<<endl;
	weight_from_comprehensiveNode_to_output = weight_from_comprehensiveNode_to_output - D_matrix * B_matrixT*train_label;
	weight_from_comprehensiveNode_to_output = bls_network::vstack(weight_from_comprehensiveNode_to_output, B_matrixT*train_label);
	increment_for_enhanceNode.position = 1;
	increment_for_enhanceNode.weight_from_mapNode_to_enhanceNode_new = weight_from_mapNode_to_enhanceNode_N;
	increment_for_enhanceNode.belta_from_mapNode_to_enhanceNode_new = belta_from_mapNode_to_enhanceNode_row_N;
	increment_nodes.push_back(increment_for_enhanceNode);
	//cout<<comprehensiveNode*weight_from_comprehensiveNode_to_output<<endl;
	//cout<<"new_enhanceNode_row: "<<enhanceNode_new.rows()<<"    "<<"new_enhanceNode_row: "<<enhanceNode_new.cols()<<endl;
	//cout<<"weight_from_comprehensiveNode_to_output"<<weight_from_comprehensiveNode_to_output.rows()<<"   "<<"weight_from_comprehensiveNode_to_output"<<weight_from_comprehensiveNode_to_output.cols()<<endl;
	double a = bls_network::calculate_accuracy(comprehensiveNode*weight_from_comprehensiveNode_to_output, train_label);
	cout << a << endl;
	cout << comprehensiveNode * weight_from_comprehensiveNode_to_output << endl;
}

void bls_network::increment_mapNode()
{
	Eigen::MatrixXd weight_from_input_to_mapNode_N;
	Eigen::MatrixXd weight_from_mapNode_to_enhanceNode_N;
	Eigen::MatrixXd belta_from_mapNode_to_enhanceNode_row_N;
	Eigen::MatrixXd belta_from_mapNode_to_enhanceNode_N;
	Eigen::MatrixXd belta_from_input_to_mapNode_row_N;
	Eigen::MatrixXd belta_from_input_to_mapNode_N;
	Eigen::MatrixXd mapNode_new;
	Eigen::MatrixXd enhanceNode_new;
	Eigen::MatrixXd VP;
	Eigen::MatrixXd B_matrixT;
	Eigen::MatrixXd C_matrix;
	Eigen::MatrixXd D_matrix;
	Eigen::MatrixXd I_matrix;
	vector<Eigen::MatrixXd> weight_from_input_vector;
	vector<Eigen::MatrixXd> weight_from_mapNode_vector;
	vector<Eigen::MatrixXd> belta_from_input_vector;
	vector<Eigen::MatrixXd> belta_from_mapNode_vector;
	vector<Eigen::MatrixXd> mapNode_vector;
	vector<Eigen::MatrixXd> enhanceNode_vector;
	vector<Eigen::MatrixXd> VP_matrix_of_mapNode_vector;
	vector<Eigen::MatrixXd> VP_matrix_of_enhanceNode_vector;

	increment increment_for_mapNode;
	weight_from_input_to_mapNode_N = Eigen::MatrixXd::Random(input_dimension, mapNode_size);
	belta_from_input_to_mapNode_row_N = Eigen::MatrixXd::Random(1, mapNode_size);
	assignment_for_belta(belta_from_input_to_mapNode_row_N, belta_from_input_to_mapNode_N, trainData_size);
	mapNode_new = train_data * weight_from_input_to_mapNode_N + belta_from_input_to_mapNode_N;
	mapNode_vector = bls_network::h_compose_matrix_to_vector(mapNode_new, mapNode_size);
	weight_from_input_vector = bls_network::h_compose_matrix_to_vector(weight_from_input_to_mapNode_N, mapNode_size);
	belta_from_input_vector = bls_network::h_compose_matrix_to_vector(belta_from_input_to_mapNode_row_N, mapNode_size);
	VP_matrix_of_mapNode_vector = bls_network::calculate_VP(mapNode_vector);
	bls_network::plus_VP(mapNode_vector, VP_matrix_of_mapNode_vector);
	bls_network::plus_VP(weight_from_input_vector, VP_matrix_of_mapNode_vector);
	bls_network::plus_VP(belta_from_input_vector, VP_matrix_of_mapNode_vector);
	mapNode_new = bls_network::h_stack_vector_to_matrix(mapNode_vector);
	weight_from_input_to_mapNode_N = bls_network::h_stack_vector_to_matrix(weight_from_input_vector);
	belta_from_input_to_mapNode_row_N = bls_network::h_stack_vector_to_matrix(belta_from_input_vector);
	mapNode_new = bls_network::sigmod(mapNode_new);
	weight_from_mapNode_to_enhanceNode_N = Eigen::MatrixXd::Random(mapNode_new.cols(), enhanceNode_size*enhanceNode_number);
	belta_from_mapNode_to_enhanceNode_row_N = Eigen::MatrixXd::Random(1, enhanceNode_size*enhanceNode_number);
	assignment_for_belta(belta_from_mapNode_to_enhanceNode_row_N, belta_from_mapNode_to_enhanceNode_N, trainData_size);
	enhanceNode_new = mapNode_new * weight_from_mapNode_to_enhanceNode_N;
	enhanceNode_new = enhanceNode_new + belta_from_mapNode_to_enhanceNode_N;
	enhanceNode_vector = bls_network::h_compose_matrix_to_vector(enhanceNode_new, enhanceNode_size);
	weight_from_mapNode_vector = bls_network::h_compose_matrix_to_vector(weight_from_mapNode_to_enhanceNode_N, enhanceNode_size);
	belta_from_mapNode_vector = bls_network::h_compose_matrix_to_vector(belta_from_mapNode_to_enhanceNode_row_N, enhanceNode_size);
	VP_matrix_of_enhanceNode_vector = bls_network::calculate_VP(enhanceNode_vector);
	bls_network::plus_VP(enhanceNode_vector, VP_matrix_of_enhanceNode_vector);
	bls_network::plus_VP(weight_from_mapNode_vector, VP_matrix_of_enhanceNode_vector);
	bls_network::plus_VP(belta_from_mapNode_vector, VP_matrix_of_enhanceNode_vector);
	enhanceNode_new = bls_network::h_stack_vector_to_matrix(enhanceNode_vector);
	weight_from_mapNode_to_enhanceNode_N = bls_network::h_stack_vector_to_matrix(weight_from_mapNode_vector);
	belta_from_mapNode_to_enhanceNode_row_N = bls_network::h_stack_vector_to_matrix(belta_from_mapNode_vector);
	enhanceNode_new = bls_network::relu(enhanceNode_new);
	//enhanceNode_new = bls_network::hstack(mapNode_new, enhanceNode_new);
	enhanceNode_new = bls_network::hstack(mapNode_new, enhanceNode_new);
	bls_network::svd(enhanceNode_new, VP);
	enhanceNode_new = enhanceNode_new * VP;
	D_matrix = pseudo_inverse_comprehensiveNode * enhanceNode_new;
	C_matrix = enhanceNode_new - comprehensiveNode * D_matrix;
	I_matrix = Eigen::MatrixXd::Ones(D_matrix.cols(), D_matrix.cols());
	if (C_matrix.all())
	{
		B_matrixT = I_matrix + D_matrix.transpose()*D_matrix;
		B_matrixT = B_matrixT.inverse();
		B_matrixT = B_matrixT * D_matrix.transpose()*pseudo_inverse_comprehensiveNode;
	}
	else
	{
		B_matrixT = calculate_pseudo_inverse(C_matrix);
	}
	comprehensiveNode = bls_network::hstack(comprehensiveNode, enhanceNode_new);
	pseudo_inverse_comprehensiveNode = pseudo_inverse_comprehensiveNode - D_matrix * B_matrixT;
	pseudo_inverse_comprehensiveNode = bls_network::vstack(pseudo_inverse_comprehensiveNode, B_matrixT);
	weight_from_comprehensiveNode_to_output = weight_from_comprehensiveNode_to_output - D_matrix * B_matrixT*train_label;
	weight_from_comprehensiveNode_to_output = bls_network::vstack(weight_from_comprehensiveNode_to_output, B_matrixT*train_label);
	increment_for_mapNode.position = 2;
	increment_for_mapNode.weight_from_mapNode_to_enhanceNode_new = weight_from_mapNode_to_enhanceNode_N;
	increment_for_mapNode.belta_from_mapNode_to_enhanceNode_new = belta_from_mapNode_to_enhanceNode_row_N;
	increment_for_mapNode.weight_from_input_to_mapNode_new = weight_from_input_to_mapNode_N;
	increment_for_mapNode.belta_from_input_to_mapNode_new = belta_from_input_to_mapNode_row_N;
	increment_for_mapNode.VP = VP;
	increment_nodes.push_back(increment_for_mapNode);
	//cout<<comprehensiveNode*weight_from_comprehensiveNode_to_output<<endl;
	double a = bls_network::calculate_accuracy(comprehensiveNode*weight_from_comprehensiveNode_to_output, train_label);
	cout << a << endl;
}

double bls_network::calculate_accuracy(Eigen::MatrixXd real, Eigen::MatrixXd predict)
{
	double accuracy_number = 0;
	double accuracy;
	real = real - predict;
	real = real.array().abs();
	for (int i = 0; i < real.rows(); i++)
	{
		if (real(i, 0) < 0.2)
		{
			accuracy_number += 1;
		}
	}
	accuracy = accuracy_number / real.rows();
	return accuracy;
}

void bls_network::predict()
{
	Eigen::MatrixXd mapNode_for_testdata;
	Eigen::MatrixXd enhanceNode_for_testdata;
	Eigen::MatrixXd comprehensiveNode_for_testdata;
	Eigen::MatrixXd belta_from_input_to_mapNode_for_testdata;
	Eigen::MatrixXd belta_from_mapNode_to_enhanceNode_for_testdata;
	assignment_for_belta(belta_from_input_to_mapNode_row, belta_from_input_to_mapNode_for_testdata, testData_size);
	assignment_for_belta(belta_from_mapNode_to_enhanceNode_row, belta_from_mapNode_to_enhanceNode_for_testdata, testData_size);
	mapNode_for_testdata = test_data * weight_from_input_to_mapNode + belta_from_input_to_mapNode_for_testdata;
	mapNode_for_testdata = bls_network::sigmod(mapNode_for_testdata);
	enhanceNode_for_testdata = mapNode_for_testdata * weight_from_mapNode_to_enhanceNode + belta_from_mapNode_to_enhanceNode_for_testdata;
	enhanceNode_for_testdata = bls_network::relu(enhanceNode_for_testdata);
	comprehensiveNode_for_testdata = bls_network::hstack(mapNode_for_testdata, enhanceNode_for_testdata);
	/*for(int i=0; i<increment_nodes.size(); i++)
	{
		increment increment_node;
		increment_node = increment_nodes[i];
		if(increment_node.position == 1)
		{
			Eigen::MatrixXd weight_from_mapNode_to_enhanceNode_N = increment_node.weight_from_mapNode_to_enhanceNode_new;
			Eigen::MatrixXd belta_from_mapNode_to_enhanceNode_N;
			Eigen::MatrixXd enhanceNode_new;
			Eigen::MatrixXd V_enhanceNode_of_testdata_new;
			Eigen::MatrixXd VP_enhanceNode_of_testdata_new;
			assignment_for_belta(increment_node.belta_from_mapNode_to_enhanceNode_new, belta_from_mapNode_to_enhanceNode_N, testData_size);
			enhanceNode_new = mapNode_for_testdata*weight_from_mapNode_to_enhanceNode_N+belta_from_mapNode_to_enhanceNode_N;
			enhanceNode_new = bls_network::relu(enhanceNode_new);
			P_1 = bls_network::svd(enhanceNode_new, V_enhanceNode_of_testdata_new);
			VP_enhanceNode_of_testdata_new = V_enhanceNode_of_testdata_new.block(0, 0, V_enhanceNode_of_testdata_new.rows(), increment_node.P_of_enhanceNode);
			//cout<<enhanceNode_new.cols()<<"   "<<VP_enhanceNode_of_testdata_new.rows()<<endl;
			enhanceNode_new = enhanceNode_new*VP_enhanceNode_of_testdata_new;
			comprehensiveNode_for_testdata = bls_network::hstack(comprehensiveNode_for_testdata, enhanceNode_new);
		}
		else
		{
			Eigen::MatrixXd weight_from_input_to_mapNode_N = increment_node.weight_from_input_to_mapNode_new;
			Eigen::MatrixXd belta_from_input_to_mapNode_N;
			Eigen::MatrixXd weight_from_mapNode_to_enhanceNode_N = increment_node.weight_from_mapNode_to_enhanceNode_new;
			Eigen::MatrixXd belta_from_mapNode_to_enhanceNode_N;
			Eigen::MatrixXd mapNode_new;
			Eigen::MatrixXd enhanceNode_new;
			Eigen::MatrixXd V_mapNode_of_testdata_new;
			Eigen::MatrixXd VP_mapNode_of_testdata_new;
			Eigen::MatrixXd V_enhanceNode_of_testdata_new;
			Eigen::MatrixXd VP_enhanceNode_of_testdata_new;
			assignment_for_belta(increment_node.belta_from_input_to_mapNode_new, belta_from_input_to_mapNode_N, testData_size);
			assignment_for_belta(increment_node.belta_from_mapNode_to_enhanceNode_new, belta_from_mapNode_to_enhanceNode_N, testData_size);
			mapNode_new = test_data*weight_from_input_to_mapNode_N + belta_from_input_to_mapNode_N;
			mapNode_new = bls_network::sigmod(mapNode_new);
			P_1 = bls_network::svd(mapNode_new, V_mapNode_of_testdata_new);
			VP_mapNode_of_testdata_new = V_mapNode_of_testdata_new.block(0, 0, V_mapNode_of_testdata_new.rows(), increment_node.P_of_mapNode);
			enhanceNode_new = mapNode_new*weight_from_mapNode_to_enhanceNode_N + belta_from_mapNode_to_enhanceNode_N;
			enhanceNode_new = bls_network::relu(enhanceNode_new);
			P_2 = bls_network::svd(enhanceNode_new, V_enhanceNode_of_testdata_new);
			VP_enhanceNode_of_testdata_new = V_enhanceNode_of_testdata_new.block(0, 0, V_enhanceNode_of_testdata_new.rows(), increment_node.P_of_enhanceNode);
			mapNode_new = mapNode_new*VP_mapNode_of_testdata_new;
			enhanceNode_new = enhanceNode_new*VP_enhanceNode_of_testdata_new;
			comprehensiveNode_for_testdata = bls_network::hstack(comprehensiveNode_for_testdata, mapNode_new);
			comprehensiveNode_for_testdata = bls_network::hstack(comprehensiveNode_for_testdata, enhanceNode_new);
		}
	}*/
	predict_label = comprehensiveNode_for_testdata * weight_from_comprehensiveNode_to_output;
	cout << predict_label << endl;
	//cout<<comprehensiveNode*weight_from_comprehensiveNode_to_output<<endl;
}
