import System.Random
import Control.Monad (replicateM)
import Data.List (transpose)

data TrainingExample = TrainingExample {
    inputs :: Input,
    target :: Output
} deriving (Show)

data Neuron = Neuron {
    weights :: [Weight]
} deriving (Show)

data Layer = Layer {
    neurons :: [Neuron]
} deriving (Show)

data Network = Network {
    layers :: [Layer]
} deriving (Show)

type Input = [Double]
type Output = [Double]
type Error = Double
type Weight = Double

quadError :: Double -> Double -> Double
quadError t o = (t - o) * (1 - o) * o

elementwiseProduct :: [Double] -> [Double] -> [Double]
elementwiseProduct = zipWith (*)

stepFunction :: Double -> Double
stepFunction x
    | x > 0 = 1
    | otherwise = 0

sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp (-x))

sigmoid' :: Double -> Double
sigmoid' x = sigmoid x * (1 - sigmoid x)

-- Initializes neuron with a given number of random weights
randomNeuron :: Int -> IO Neuron
randomNeuron numWeights = do
    ws <- replicateM numWeights (randomRIO (-1, 1) :: IO Double)
    return $ Neuron ws

feedForward :: Input -> Network -> Output
feedForward i (Network []) = i
feedForward i (Network (l1:l2)) = feedForward i' $ Network l2
    where i' = map (sigmoid . sum . elementwiseProduct i . weights) $ neurons l1

scanForward :: Input -> Network -> [Output]
scanForward i (Network ls) = scanl (\lMinus1 l -> map (sigmoid . sum . elementwiseProduct lMinus1 . weights) $ neurons l) i ls

neuronError :: Error -> (Output, Neuron) -> [Error]
neuronError e (os, Neuron ws) = zipWith (\o w -> (w * e * (1 - o) * o)) os ws

layerError :: [Error] -> [(Output, Neuron)] -> [Error]
layerError errors outputs = map sum . transpose $ zipWith neuronError errors outputs

adjustNeuron :: Error -> (Output, Neuron) -> Neuron
adjustNeuron error (output, Neuron ws) = Neuron updatedWeights
    where delta = map (* error) output
          updatedWeights = zipWith (+) delta ws

adjustedLayer :: [Error] -> [(Output, Neuron)] -> Layer
adjustedLayer errors outputs = Layer $ zipWith adjustNeuron errors outputs

layerOutputs :: Input -> Network -> [(Output, Layer)]
layerOutputs i n = zip outputs ls
    where outputs = tail . reverse $ scanForward i n
          ls = reverse . layers $ n

backPropagate :: [([Error], Layer)] -> [[(Output, Neuron)]] -> Network
backPropagate acc [] = Network . reverse . map snd . tail $ acc
backPropagate acc (l1:l2) = backPropagate (acc ++ [(el1, l1')]) l2
    where elMinus1 = fst . last $ acc
          el1 = layerError elMinus1 l1
          l1' = adjustedLayer elMinus1 l1

groupOutputs :: (Output, Layer) -> [(Output, Neuron)]
groupOutputs (outputs, Layer ns) = map ((,) outputs) ns

train :: TrainingExample -> Network -> Network
train t n = backPropagate [(error, Layer [])] outputs
    where activations = scanForward (inputs t) n
          error = zipWith quadError (target t) $ last activations
          outputs = zipWith (curry groupOutputs) (tail . reverse $ activations) (reverse . layers $ n)

