// Inspired by https://gist.github.com/karpathy/d4dee566867f8291f086 and https://towardsdatascience.com/recurrent-neural-networks-rnns-3f06d7653a85.

package rnn

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type RNN struct {
	hiddenSize   int
	inputSize    int
	seqLength    int
	learningRate int

	u, v, w *mat.Dense
	b, c    *mat.VecDense
}

func New(hiddenSize, inputSize, seqLength, learningRate int) *RNN {
	num1 := math.Sqrt(1.0 / float64(inputSize))
	num2 := math.Sqrt(1.0 / float64(hiddenSize))

	uNums := make([]float64, hiddenSize*inputSize)
	for i := range uNums {
		uNums[i] = -num1 + rand.Float64()*(2*num1)
	}
	u := mat.NewDense(hiddenSize, inputSize, uNums)

	vNums := make([]float64, inputSize*hiddenSize)
	for i := range vNums {
		vNums[i] = -num2 + rand.Float64()*(2*num2)
	}
	v := mat.NewDense(inputSize, hiddenSize, vNums)

	wNums := make([]float64, hiddenSize*hiddenSize)
	for i := range vNums {
		wNums[i] = -num2 + rand.Float64()*(2*num2)
	}
	w := mat.NewDense(hiddenSize, hiddenSize, vNums)

	return &RNN{
		hiddenSize:   hiddenSize,
		inputSize:    inputSize,
		seqLength:    seqLength,
		learningRate: learningRate,

		u: u,
		v: v,
		w: w,
		// Allocate zero-ed cells.
		b: mat.NewVecDense(hiddenSize, nil),
		c: mat.NewVecDense(inputSize, nil),
	}
}

type reader func() ([]int, []int, bool)

func (n *RNN) Train(read reader) {
	var (
		threshold   = 0.01
		smooth_loss = -math.Log(1.0 / float64(n.inputSize))
		hprev       = mat.NewVecDense(n.hiddenSize, nil)

		restart         bool
		inputs, targets []int

		i int
	)

	for smooth_loss > threshold {
		if restart {
			hprev = mat.NewDense(n.hiddenSize, 1, nil)
		}

		inputs, targets, restart = read()
		xs, hs, ps := forward()
	}
}

func (n *RNN) forward(inputs []int, hprev *mat.VecDense) (xs, hs, ycap []*mat.VecDense) {
	var (
		os []*mat.VecDense

		vec1, vec2 *mat.VecDense
	)
	hs[0].CopyVec(hprev)

	for t, input := range inputs {
		xs[t] = mat.NewVecDense(n.inputSize, nil)
		xs[t].SetVec(input, 1) // One hot encode.

		vec1.MulVec(n.u, xs[t])
		vec2.MulVec(n.w, hs[t])

		vec1.AddVec(vec1, vec2)
		vec1.AddVec(vec1, n.b)

		apply(vec1, func(n float64) float64 {
			return math.Tanh(n)
		})
		hs[t].CopyVec(vec1)

		vec1.MulVec(n.v, hs[t])
		os[t].CopyVec(vec1)

		ycap[t].CopyVec(os[t])
		softmax(ycap[t])
	}

	return xs, hs, ycap
}

func softmax(v *mat.VecDense) {
	var (
		m   = v.T()
		max = mat.Max(m)
	)

	apply(v, func(n float64) float64 {
		return n - max
	})

	d := mat.Dense{}
	d.Exp(v.T())

	v = mat.VecDenseCopyOf(d.ColView(1))
	apply(v, func(n float64) float64 {
		return n / mat.Sum(d.T())
	})
}

func (n *RNN) backward(xs, hs, ps []*mat.VecDense, targets []int) {
	var (
		dU     = mat.NewDense(n.hiddenSize, n.inputSize, nil)
		dV     = mat.NewDense(n.inputSize, n.hiddenSize, nil)
		dW     = mat.NewDense(n.hiddenSize, n.hiddenSize, nil)
		dB     = mat.NewVecDense(n.hiddenSize, nil)
		dC     = mat.NewVecDense(n.inputSize, nil)
		dHNext = mat.NewVecDense(n.hiddenSize, nil)

		dy, vec1 *mat.VecDense
	)

	for t := n.seqLength; t >= 0; t-- {
		dy.CopyVec(hs[t])
		dy.SetVec(targets[t], dy.AtVec(targets[t])-1)

		vec1.MulVec()
	}
}

func apply(v *mat.VecDense, a func(n float64) float64) {
	for i := 0; i < v.Len(); i++ {
		v.SetVec(1, a(v.AtVec(i)))
	}
}
