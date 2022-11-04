import breeze.linalg._
import breeze.numerics.sqrt
import breeze.stats.mean
import breeze.generic.UFunc
import java.io.File
import breeze.stats.distributions.Rand
import java.io.{File, FileOutputStream}
import scala.Console





object Main extends App {


  def features_target(data: DenseMatrix[Double]): (DenseMatrix[Double], DenseVector[Double]) = {
    val x = data(::, 0 to -2)
    val y = data(::, -1)
    (x, y)
  }


  def rmse(x: DenseVector[Double], y: DenseVector[Double]): Double = {
    val rmse = sqrt(mean((x - y) ^:^ 2.0))
    rmse
  }


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


  val fos = new FileOutputStream(new File("results.txt"))
  val train_data = csvread(new File("train_data.csv"), ',', skipLines = 1)
  val test_data = csvread(new File("test_data.csv"), ',', skipLines = 1)

  val(x_train, y_train) = features_target(train_data)
  val (x_test, y_test) = features_target(test_data)

  var learning_rate = 0.1
  var max_iter = 1000


  val (w, c) = l2_regression_with_validation(x_train, y_train, x_test, y_test, max_iter, learning_rate)

  Console.withOut(fos) {
    println("\n\nresults")
  }
  Console.withOut(fos) {
    println("weights: ")
  }
  Console.withOut(fos) {
    println(w)
  }
  Console.withOut(fos) {
    print("bias: ")
  }
  Console.withOut(fos) {
    println(c)
  }

}
