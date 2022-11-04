
import breeze.linalg._
import breeze.numerics.sqrt
import breeze.stats.mean
import breeze.generic.UFunc
import java.io.File
import breeze.stats.distributions.Rand
import java.io.{File, FileOutputStream}
import scala.Console

val fos = new FileOutputStream(new File("results.txt"))


val train_data = csvread(new File("train_data.csv"), ',', skipLines=1)
val test_data = csvread(new File("/home/roman/Mail_MADE/semestr3/mlbd/hw3/scala_lr/test_data.csv"), ',', skipLines=1)

def features_target(data: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double]) = {
  val x = data(::, 0 to -2)
  val y = data(::, -1)
  (x,y)
}


def rmse(x:DenseVector[Double], y:DenseVector[Double]): Double = {
  val rmse = sqrt(mean((x - y) ^:^ 2.0))
  rmse
}


5 == 2



def l2_regression_with_validation(x_train:DenseMatrix[Double],
                                  y_train:DenseVector[Double],
                                  x_test:DenseMatrix[Double],
                                  y_test:DenseVector[Double],
                                  max_iter:Int,
                                  lr: Double): (DenseVector[Double], Double) = {

  var prev_w = DenseVector.ones[Double](x_train.cols) * 1000.0
  //var prev_c = DenseVector.ones[Double](1) * 1000.0
  var next_w = DenseVector.rand(x_train.cols, Rand.uniform)
  val rand = new scala.util.Random
  var prev_c = 1000.0
  var next_c = rand.nextDouble()
  //var next_c = DenseVector.rand(1, Rand.uniform)
  var learning_rate = lr

  for(i <- 0 to max_iter){
    if (rmse(prev_w, next_w) > 1e-6 && (prev_c - next_c).abs > 1e-6){
      // calc gradients and new weights
      val y_train_pred = x_train * next_w + next_c
      val dw =  - x_train.t * (y_train - y_train_pred) / x_train.rows.toDouble
      val dc =  - mean(y_train - y_train_pred)
      //val diff = (y_train_pred - y_train) ^:^ 2.0
      prev_w = next_w
      prev_c = next_c
      next_w = prev_w - learning_rate * dw
      next_c = prev_c - learning_rate * dc

      // calc score on test dataset
      val y_test_pred = x_test * next_w + next_c
      val rmse_score = rmse(y_test_pred, y_test)
      Console.withOut(fos) {println("iter " + i.toString)}
      Console.withOut(fos) {print("rmse on test dataset: " + rmse_score.toString + "\n")}


      //if (i %100 == 0){
      //  learning_rate = learning_rate / 1.5
      //}

    } else {
      return (next_w, next_c)
    }


  }

  (next_w, next_c)
}

var learning_rate = 0.1
var max_iter = 1000


val (x_train, y_train) = features_target(train_data)
val (x_test, y_test) = features_target(test_data)


val (w, c) = l2_regression_with_validation(x_train, y_train, x_test, y_test, max_iter, learning_rate)

Console.withOut(fos) {println("results")}
Console.withOut(fos) {println("weights: ")}
Console.withOut(fos) {println(w)}
Console.withOut(fos) {print("bias: ")}
Console.withOut(fos) {println(c)}

// [27.59964018, 63.71927988, 54.22098987, 94.63536122, 59.96468424] , 0.5

//DenseVector(27.597066242633723, 63.70576157664248, 54.22023377936278, 94.61848280276865, 59.95363213476943)
//0.5002212401801146





//println("privet " + "123")



//5.0 / 2
//
//
//val init_w = DenseVector.rand(x_train.cols, Rand.uniform)
//
//
////println(b)
//
//val xx = randomDouble()
//
//xx
//
//
//
//
//
//
//
//
//y_train
//
//
//
//
//
//
//val x_train = train_data(::, 0 to -2)
//
//
//println(train_data(0 to 10, ::))
//
//val arr = DenseMatrix((1,2,3), (4,5,6))
//
//val vec1 = DenseVector(1.0, 2.0, 3.0)
//val vec2= DenseVector(4,5,6)
//val m = mean(convert(vec1, Double))
//
//val m = mean(vec1 ^:^ 2.0)
//m
//
//
//val vec3 = vec1 + 3.0
//
//vec3
////val v2 = (vec1 - vec2 * 2) ^:^ 2
////v2
//
//
////val v3 = op(v2,2)
//
//arr(::, -1)
//
//
////vec
//
//
//
////val r = arr * vec
//
////r
//
//
//
//def add(a: Int, b: Int): Int = {
//  val i: Int = a + b
//  return i
//}
//
//println(add(4,6))