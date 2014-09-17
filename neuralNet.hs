{-# LANGUAGE BangPatterns #-}
import System.Random
import Control.Monad (replicateM)
import qualified Data.Vector as V

data TrainingVector = TrainingVector {
    inputs :: Input,
    output :: Output
} deriving (Show)

data Neuron = Neuron {
    weights :: V.Vector Double
} deriving (Show)

data Layer = Layer {
    neurons :: [Neuron]
} deriving (Show)

data Network = Network {
    layers :: [Layer]
} deriving (Show)

type Input = V.Vector Double
type Output = V.Vector Double

feedForward :: Input -> Network -> Output
feedForward i (Network []) = i
feedForward i (Network (l1:l2)) = feedForward i' $ Network l2
    where i' = V.fromList $ map (sigmoid . V.sum . elementwiseProduct i . weights) $ neurons l1

scanForward :: Input -> Network -> [Output]
scanForward i (Network ls) = scanl (\lMinus1 l -> V.fromList $ map (sigmoid . V.sum . elementwiseProduct lMinus1 . weights) $ neurons l) i ls

elementwiseProduct :: V.Vector Double -> V.Vector Double -> V.Vector Double
elementwiseProduct = V.zipWith (*)

stepFunction :: Double -> Double
stepFunction x
    | x > 0 = 1
    | otherwise = 0

sigmoid :: Double -> Double
sigmoid !x = 1 / (1 + exp (-x))

sigmoid' :: Double -> Double
sigmoid' !x = sigmoid x * (1 - sigmoid x)

randomNeuron :: Int -> IO Neuron
randomNeuron numWeights = do
    ws <- replicateM numWeights (randomRIO (-1, 1) :: IO Double)
    return . Neuron . V.fromList $ ws

neuronFromWeights :: [Double] -> Neuron
neuronFromWeights = Neuron . V.fromList

toV :: [Double] -> V.Vector Double
toV = V.fromList
