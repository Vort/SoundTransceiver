using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SoundReceiver
{
    class SignalException : Exception
    {
        public SignalException(string message)
            : base(message)
        {
        }
    }

    class Detector
    {
        const int sampleRate = 44100;

        public double bitrate = 100;
        public double carrierFreq = 1000;

        const double maxFreqDeviation = 0.005;

        sbyte[] equalizerTrainSequence = new sbyte[]
        {
            1, 1, 1, 1, 1, -1, 1, -1,
            -1, -1, 1, -1, 1, 1, 1, 1,
            1, -1, 1, -1, -1, 1, 1, -1,
            -1, -1, -1, 1, -1, -1, -1, -1
        };
        const int sendRrcBitCount = 64;
        const int recvRrcBitCount = 4;
        const double rrcBeta = 0.8;

        const int payloadSizeLsbSize = 3;

        Fourier fourier;

        public Detector()
        {
            fourier = new Fourier();
        }

        public byte[] Detect(short[] signal, List<double> bitLevels, out double? snr, out double? mer)
        {
            if (signal.Length == 0)
                throw new SignalException("No data");

            bool debug = false;

            double[] d1 = SignalStoD(signal);

            double bandwidth = (1 + rrcBeta) * bitrate;
            double freq = carrierFreq;
            double freq1 = (freq - bandwidth / 2.0) * (1.0 - maxFreqDeviation);
            double freq2 = (freq + bandwidth / 2.0) * (1.0 + maxFreqDeviation);
            int f1size = (int)(4.0 * sampleRate / (freq2 - freq1)) | 1;
            double[] f1 = Mul(MakeBandPassFilter(freq1, freq2, f1size), MakeHannWindow(f1size));
            double[] d2 = Convolve(f1, d1);
            if (debug) SaveWav("d2.wav", SignalDtoS(d2));

            double bitLen = sampleRate / bitrate;

            int integrateBitLen1 = Convert.ToInt32(bitLen * 2);
            int integrateBitLen2 = Convert.ToInt32(bitLen * 32);

            double[] d21 = Abs(d2);
            double[] d22 = Integrate(d21, integrateBitLen1);
            if (debug) SaveWav("d22.wav", SignalDtoS(d22));
            double[] d23 = Integrate(d21, integrateBitLen2);
            if (debug) SaveWav("d23.wav", SignalDtoS(d23));

            double[] d24 = new double[d22.Length - integrateBitLen2];
            double[] d25 = new double[d22.Length - integrateBitLen2];
            MinMax(d22, integrateBitLen2, d24, d25);

            if (debug) SaveWav("d24.wav", SignalDtoS(d24));
            if (debug) SaveWav("d25.wav", SignalDtoS(d25));


            double noiseLevel = 0.0;
            int signalStartCoarse = 0;
            int signalEndCoarse = 0;
            for (int i = integrateBitLen2; i < d24.Length; i++)
            {
                if ((signalStartCoarse == 0) &&
                    (d24[i] > 2.0 * d23[i]))
                {
                    noiseLevel = d23[i - integrateBitLen1];
                    signalStartCoarse = i;
                }
                if ((signalStartCoarse != 0) &&
                    (signalEndCoarse == 0) &&
                    (d25[i] <= 2.0 * noiseLevel))
                {
                    signalEndCoarse = i;
                    break;
                }
            }

            if (signalStartCoarse == 0)
                throw new SignalException("Signal start is not found");
            if (signalEndCoarse == 0)
                throw new SignalException("Signal end is not found");

            signalStartCoarse -= Convert.ToInt32(bitLen);
            signalEndCoarse += Convert.ToInt32(bitLen);
            if (signalStartCoarse < 0)
                signalStartCoarse = 0;
            if (signalEndCoarse > d24.Length)
                signalEndCoarse = d24.Length;

            int signalCenter = (signalStartCoarse + signalEndCoarse) / 2;

            int signalStartFine = 0;
            int signalEndFine = 0;
            double signalAvgMin = d24[signalCenter - integrateBitLen2 / 2];
            for (int i = signalStartCoarse; i < signalEndCoarse; i++)
            {
                if ((signalStartFine == 0) &&
                    (d24[i] > 0.25 * signalAvgMin))
                {
                    signalStartFine = i;
                }
                if ((signalStartFine != 0) &&
                    (signalEndFine == 0) &&
                    (d25[i] < 0.5 * signalAvgMin))
                {
                    signalEndFine = i;
                    break;
                }
            }
            if (signalEndFine == 0)
                signalEndFine = signalEndCoarse;

            if (signalStartFine == 0)
                throw new SignalException("Signal fine start is not found");

            // Adjustments are needed to correct shifts, which were made by integration
            // Signal positions are offsets from d1 of the peaks for first and last bits
            signalStartFine = Convert.ToInt32(signalStartFine + f1size / 2 + 0.1 * bitLen);
            signalEndFine = Convert.ToInt32(signalEndFine + f1size / 2 - 1.7 * bitLen);

            // SNR calculation layout:
            // 16 bits of noise, 32 + 4 bits skipped, 24 bits of train sequence
            int noisePowerStart = Convert.ToInt32(signalStartFine - (32 + 16) * bitLen - f1size / 2);
            int trainPowerStart = Convert.ToInt32(signalStartFine + 4 * bitLen - f1size / 2);
            int noisePowerEnd = Convert.ToInt32(noisePowerStart + 16 * bitLen);
            int trainPowerEnd = Convert.ToInt32(trainPowerStart + 24 * bitLen);
            double noisePower = 0.0;
            double trainPower = 0.0;
            if (noisePowerStart < 0 || trainPowerEnd >= d2.Length)
                throw new SignalException("Not enough samples for SNR calculation");
            for (int i = noisePowerStart; i < noisePowerEnd; i++)
                noisePower += d2[i] * d2[i];
            for (int i = trainPowerStart; i < trainPowerEnd; i++)
                trainPower += d2[i] * d2[i];
            noisePower /= 16;
            trainPower /= 24;
            if (noisePower != 0.0 && trainPower > noisePower)
                snr = 10 * Math.Log10((trainPower - noisePower) / noisePower);
            else
                snr = null;

            double[] d2s = new double[signalEndFine - signalStartFine];
            for (int i = 0; i < d2s.Length; i++)
            {
                double d2v = d2[i + signalStartFine - f1size / 2];
                d2s[i] = d2v * d2v;
            }

            if (debug) SaveWav("d2s.wav", SignalDtoS(Normalize(d2s)));

            freq1 = (freq - freq * maxFreqDeviation) * 2;
            freq2 = (freq + freq * maxFreqDeviation) * 2;
            double estFreq = EstimateFrequency(d2s, freq1, freq2) / 2;


            double estBitrate = bitrate * estFreq / freq;
            int decFactor = (int)(sampleRate / estBitrate / 3);
            int decFilterSize = decFactor == 1 ? 0 : (int)(8 * sampleRate / estBitrate) | 1;
            double decSampleRate = (double)sampleRate / decFactor;
            double estBitlen = decSampleRate / estBitrate;



            double[] trainSignal = MakePulsesShape(
                equalizerTrainSequence, decSampleRate, estBitrate, rrcBeta, sendRrcBitCount);
            if (debug) SaveWav("trainSignal.wav", SignalDtoS(Mul(trainSignal, 0.67)), Convert.ToInt32(decSampleRate));

            double[] rrc = Mul(MakeRootRaisedCosine(
                (int)(recvRrcBitCount * estBitlen) | 1,
                estBitlen, rrcBeta), 1 / estBitlen);

            if (debug) SaveWav("rrc.wav", SignalDtoS(rrc));



            int deltaRange = Convert.ToInt32(estBitlen * 2);
            //int deltaRange = 0;

            int eqSize = (int)(10 * estBitlen) | 1;
            Complex[] d3 = new Complex[signalEndFine - signalStartFine +
                (rrc.Length + eqSize + deltaRange * 2) * decFactor + decFilterSize];
            int d3offset = signalStartFine - deltaRange * decFactor -
                (rrc.Length + eqSize) * decFactor / 2 - decFilterSize / 2;
            double phaseScale = 2 * Math.PI * estFreq / sampleRate;
            for (int i = 0; i < d3.Length; i++)
            {
                double d1v = d1[d3offset + i];
                double phase = i * phaseScale;
                d3[i] = new Complex(
                    d1v * Math.Cos(phase),
                    -d1v * Math.Sin(phase));
            }

            if (decFactor != 1)
            {
                double[] flt = MakeLowPassFilter(sampleRate / 2.0 / decFactor, decFilterSize);
                d3 = Decimate(flt, d3, decFactor);
            }

            if (debug)
            {
                double[] d3i = d3.Select(x => x.Real).ToArray();
                double[] d3q = d3.Select(x => x.Imaginary).ToArray();

                double maxAbsI = MaxAbs(d3i);
                double maxAbsQ = MaxAbs(d3q);
                double maxAbsIQ = Math.Max(maxAbsI, maxAbsQ);

                SaveWav("d3i.wav", SignalDtoS(Mul(d3i, 1.0 / maxAbsIQ)), (int)decSampleRate);
                SaveWav("d3q.wav", SignalDtoS(Mul(d3q, 1.0 / maxAbsIQ)), (int)decSampleRate);
            }

            int trainOffset1 = rrc.Length / 2;
            int trainOffset2 = (int)(sendRrcBitCount * estBitlen) / 2;
            int trainSize = Convert.ToInt32(estBitlen * (equalizerTrainSequence.Length - 1));

            int bestDelta = 0;
            Complex[] bestEq = null;
            double ajMin = double.MaxValue;
            Complex[,] s = Transpose(ToComplex(
                trainSignal.Skip(trainOffset2).Take(trainSize).ToArray()));
            Complex[,] sct = ConjugateTranspose(s);
            Complex scts = Mul(sct, s)[0, 0];

            if (trainOffset1 + deltaRange * 2 + eqSize - 1 + trainSize > d3.Length)
                throw new SignalException("Equalizer construction failed");

            Parallel.For(0, deltaRange * 2 + 1, d =>
            {
                Complex[,] r = ToeplitzMatrix(
                    d3.Skip(trainOffset1 + d + eqSize - 1).Take(trainSize).ToArray(),
                    d3.Skip(trainOffset1 + d).Take(eqSize).Reverse().ToArray());
                Complex[,] rct = ConjugateTranspose(r);
                Complex[,] rctr = Mul(rct, r);
                Complex[,] rcts = Mul(rct, s);
                Complex[,] sctr = Mul(sct, r);
                Complex[,] f = Mul(Inverse(rctr), rcts);
                double aj = Complex.Abs(scts - Mul(sctr, f)[0, 0]);
                lock (s)
                {
                    if (aj < ajMin)
                    {
                        ajMin = aj;
                        bestEq = GetColumn(f, 0);
                        bestDelta = d;
                    }
                }
            });

            var d3f = Convolve(bestEq, d3);

            double[] d4 = null;
            int debugSampleRate = 0;
            const double d3scale = 0.25;
            const int debugInterpolationFactor = 8;
            if (debug)
            {
                double[] debugRrc = Mul(MakeRootRaisedCosine(
                    (rrc.Length - 1) * debugInterpolationFactor + 1,
                    estBitlen * debugInterpolationFactor,
                    rrcBeta), 1 / estBitlen);

                var d3if = d3f.Select(x => x.Real).ToArray();
                var d3qf = d3f.Select(x => x.Imaginary).ToArray();
                SaveWav("d3if.wav", SignalDtoS(Mul(d3if, d3scale)), (int)decSampleRate);
                SaveWav("d3qf.wav", SignalDtoS(Mul(d3qf, d3scale)), (int)decSampleRate);

                debugSampleRate = (int)(decSampleRate * debugInterpolationFactor);
                var d3if2 = Convolve(debugRrc, StuffZeroes(d3if, debugInterpolationFactor));
                var d3qf2 = Convolve(debugRrc, StuffZeroes(d3qf, debugInterpolationFactor));
                SaveWav("d3if2.wav", SignalDtoS(Mul(d3if2, d3scale)), debugSampleRate);
                SaveWav("d3qf2.wav", SignalDtoS(Mul(d3qf2, d3scale)), debugSampleRate);

                d4 = new double[d3if2.Length];
            }

            var bits = new List<bool>();
            var bitMER = new List<double>();

            for (double tf = bestDelta; tf < d3f.Length - rrc.Length + 1; tf += estBitlen)
            {
                int ti = (int)tf;
                double ts = tf - ti;

                double[] rrcSubsample = MakeRootRaisedCosine(
                    (int)(recvRrcBitCount * estBitlen) | 1,
                    estBitlen, rrcBeta, ts);

                Complex sample = Convolve(rrcSubsample, d3f, ti) / estBitlen;

                double di = 1.0 - Math.Abs(sample.Real);
                double dq = 0.0 - sample.Imaginary;
                bitMER.Add(di * di + dq * dq);

                bits.Add(sample.Real > 0.0);
                bitLevels.Add(Math.Abs(sample.Real));

                if (debug)
                {
                    int ti2 = Convert.ToInt32(tf * debugInterpolationFactor);
                    if (ti2 == d4.Length)
                        ti2--;
                    d4[ti2] = sample.Real;
                }
            }

            if (debug) SaveWav("d4.wav", SignalDtoS(Mul(d4, d3scale)), debugSampleRate);

            if (bits.Count < equalizerTrainSequence.Length + payloadSizeLsbSize)
                throw new SignalException("Wrong packet format");

            bool[] plszlsbb = bits.Skip(equalizerTrainSequence.Length).Take(payloadSizeLsbSize).ToArray();
            int plszlsb = 0;
            for (int i = 0; i < payloadSizeLsbSize; i++)
                plszlsb |= plszlsbb[i] ? (1 << payloadSizeLsbSize - i - 1) : 0;

            int plszEst = (bits.Count - equalizerTrainSequence.Length - payloadSizeLsbSize) / 8;
            int plsz = plszEst;
            int lsbMask = (1 << payloadSizeLsbSize) - 1;
            int plszEstLsb = plszEst & lsbMask;
            if (plszEstLsb != plszlsb)
            {
                plsz &= ~lsbMask;
                plsz |= plszlsb;
                if (plszEstLsb < plszlsb)
                {
                    plsz -= 1 << payloadSizeLsbSize;
                    if (plsz < 0)
                        throw new SignalException("Wrong payload size");
                }
            }

            int skippedBitCount = bits.Count -
                equalizerTrainSequence.Length - payloadSizeLsbSize - plsz * 8;
            bits.RemoveRange(bitLevels.Count - skippedBitCount, skippedBitCount);
            bitMER.RemoveRange(bitLevels.Count - skippedBitCount, skippedBitCount);
            bitLevels.RemoveRange(bitLevels.Count - skippedBitCount, skippedBitCount);
            bitMER.RemoveRange(0, equalizerTrainSequence.Length);
            bitLevels.RemoveRange(0, equalizerTrainSequence.Length);

            mer = Math.Log10(bitMER.Count() / bitMER.Sum()) * 10;

            if (!bits.Take(equalizerTrainSequence.Length).
                SequenceEqual(equalizerTrainSequence.Select(b => b == 1)))
            {
                throw new SignalException("Signal start marker is not found");
            }

            bits.RemoveRange(0, equalizerTrainSequence.Length + payloadSizeLsbSize);

            byte[] data = new byte[plsz];
            for (int i = 0; i < plsz * 8; i++)
                data[i / 8] |= (byte)((bits[i] ? 1 : 0) << (7 - i % 8));

            return data;
        }

        double EstimateFrequency(double[] signal, double freq1, double freq2)
        {
            var n = 1 << (int)Math.Log(signal.Length - 1, 2) + 1;
            var signalp2c = fourier.Transform(
                PadLeft(n * 4 - signal.Length, signal)).ToArray();

            double[] signalp2a = new double[signalp2c.Length];
            for (int k = 0; k < signalp2a.Length; k++)
            {
                signalp2a[k] = signalp2c[k].Real * signalp2c[k].Real +
                    signalp2c[k].Imaginary * signalp2c[k].Imaginary;
            }

            var maxPos = 0;
            var maxVal = double.MinValue;
            var x1 = (int)(freq1 / sampleRate * signalp2a.Length);
            var x2 = (int)(freq2 / sampleRate * signalp2a.Length);
            for (int x = x1; x < x2; x++)
            {
                if (signalp2a[x] > maxVal)
                {
                    maxVal = signalp2a[x];
                    maxPos = x;
                }
            }

            var vm1 = signalp2a[maxPos - 1];
            var vz = signalp2a[maxPos];
            var vp1 = signalp2a[maxPos + 1];

            var u = (64.0 * signal.Length * signal.Length / n) /
                (Math.Pow(Math.PI, 5) + 32 * Math.PI);
            var v = u * Math.Pow(Math.PI, 2) / 4;
            var wa = (vp1 - vm1) / (u * (vp1 + vm1) + v * vz);

            return ((double)maxPos / n + wa) * sampleRate / 4;
        }

        double[] MakePulsesShape(sbyte[] symbols,
            double sampleRate, double bitrate, double rrcBeta, int rrcBitCount)
        {
            double bitDuration = sampleRate / bitrate;
            int rrcLen = (int)(rrcBitCount * bitDuration) | 1;
            int signalLen = (int)((symbols.Length - 1) * bitDuration + 1);

            double[] signal = new double[signalLen + rrcLen];
            for (int i = 0; i < symbols.Length; i++)
            {
                double bitPosition = i * bitDuration;
                int bitPositionI = (int)Math.Floor(bitPosition);
                double bitPositionF = bitPosition - bitPositionI;
                for (int j = 0; j < rrcLen; j++)
                {
                    signal[bitPositionI + j] += RootRaisedCosine(
                        j - rrcLen / 2 - bitPositionF, bitDuration, rrcBeta) * symbols[i];
                }
            }
            return signal;
        }

        double RootRaisedCosine(double t, double bitDuration, double beta)
        {
            const double epsilon = 1e-12;
            double fourbeta = 4.0 * beta;
            if (Math.Abs(t) < epsilon)
            {
                return 1.0 + beta * (4.0 / Math.PI - 1.0);
            }
            else if (Math.Abs(Math.Abs(t) - bitDuration / fourbeta) > epsilon)
            {
                double f = t / bitDuration;
                return (
                    Math.Sin((Math.PI * f) * (1.0 - beta)) +
                    (fourbeta * f) *
                    Math.Cos((Math.PI * f) * (1.0 + beta))) /
                    ((Math.PI * f) *
                    (1.0 - fourbeta * fourbeta * f * f));
            }
            else
            {
                return (beta / Math.Sqrt(2.0)) *
                    ((1.0 + 2.0 / Math.PI) * Math.Sin(Math.PI / fourbeta) +
                    (1.0 - 2.0 / Math.PI) * Math.Cos(Math.PI / fourbeta));
            }
        }

        double[] MakeRootRaisedCosine(int filterSize,
            double bitDuration, double beta, double subsampleShift = 0.0)
        {
            double[] rootRaisedCosine = new double[filterSize];
            for (int i = 0; i < filterSize; i++)
            {
                rootRaisedCosine[i] = RootRaisedCosine(
                    i - filterSize / 2 - subsampleShift, bitDuration, beta);
            }
            return rootRaisedCosine;
        }

        Complex[] Convolve(Complex[] kernel, Complex[] signal)
        {
            if (signal.Length < kernel.Length)
                return new Complex[0];

            Complex[] r = new Complex[signal.Length - kernel.Length + 1];

            int blockSize = 512;
            int blockCount = (r.Length + blockSize - 1) / blockSize;

            Parallel.For(0, blockCount, i =>
            {
                int startSample = i * blockSize;
                int xendSample = blockSize;
                if (i == blockCount - 1)
                    xendSample = r.Length - blockSize * i;
                xendSample += startSample;
                for (int j = startSample; j < xendSample; j++)
                {
                    Complex sum = Complex.Zero;
                    for (int k = 0, l = kernel.Length - 1; k < kernel.Length; k++, l--)
                        sum += kernel[l] * signal[j + k];
                    r[j] = sum;
                }
            });
            return r;
        }

        double[] Convolve(double[] kernel, double[] signal)
        {
            if (signal.Length < kernel.Length)
                return new double[0];

            double[] r = new double[signal.Length - kernel.Length + 1];

            int blockSize = 512;
            int blockCount = (r.Length + blockSize - 1) / blockSize;

            Parallel.For(0, blockCount, i =>
            {
                int startSample = i * blockSize;
                int xendSample = blockSize;
                if (i == blockCount - 1)
                    xendSample = r.Length - blockSize * i;
                xendSample += startSample;
                for (int j = startSample; j < xendSample; j++)
                {
                    double sum = 0;
                    for (int k = 0, l = kernel.Length - 1; k < kernel.Length; k++, l--)
                        sum += kernel[l] * signal[j + k];
                    r[j] = sum;
                }
            });
            return r;
        }

        Complex Convolve(double[] kernel, Complex[] signal, int signalOffset)
        {
            Complex r = Complex.Zero;
            for (int i = 0; i < kernel.Length; i++)
                r += kernel[i] * signal[signalOffset + i];
            return r;
        }

        Complex[] Decimate(double[] filter, Complex[] signal, int factor)
        {
            if (factor <= 0)
                throw new Exception();
            if (factor == 1)
                return signal;
            if (signal.Length == 0)
                return signal;

            Complex[] result = new Complex[(signal.Length - filter.Length) / factor];
            for (int i = 0; i < result.Length; i++)
            {
                Complex sum = Complex.Zero;
                for (int k = 0; k < filter.Length; k++)
                    sum += filter[k] * signal[i * factor + k];
                result[i] = sum;
            }
            return result;
        }

        double[] StuffZeroes(double[] signal, int factor)
        {
            if (factor <= 0)
                throw new Exception();
            if (factor == 1)
                return signal;
            if (signal.Length == 0)
                return signal;

            int ptr = 0;
            double[] result = new double[signal.Length * factor];
            for (int i = 0; i < signal.Length; i++)
            {
                result[ptr++] = signal[i];
                for (int j = 1; j < factor; j++)
                    result[ptr++] = 0.0;
            }
            return result;
        }

        double[] MakeBandPassFilter(double freq1, double freq2, int sampleCount)
        {
            if (freq1 <= 0)
                return MakeLowPassFilter(freq2, sampleCount);
            else
            {
                double[] d1 = MakeLowPassFilter(freq2, sampleCount);
                double[] d2 = MakeLowPassFilter(freq1, sampleCount);
                double[] d3 = Sub(d1, d2);
                return d3;
            }
        }

        double Sinc(double x)
        {
            if (x == 0.0)
                return 1.0;
            return Math.Sin(x) / x;
        }

        double[] MakeLowPassFilter(double cutoffFreq, int sampleCount)
        {
            double w = 2 * Math.PI * (cutoffFreq / sampleRate);
            double a = w / Math.PI;

            int n = sampleCount / 2;

            double[] ir = new double[sampleCount];

            for (int i = 0; i < n + 1; i++)
            {
                if (i == 0)
                    ir[n] = a;
                else
                {
                    double V = a * Sinc(i * w);
                    ir[n + i] = V;
                    ir[n - i] = V;
                }
            }
            return ir;
        }

        double[] PadLeft(int sampleCount, double[] signal)
        {
            return new double[sampleCount].Concat(signal).ToArray();
        }

        double[] MakeHannWindow(int sampleCount)
        {
            // Only non-zero values are used
            double k = 2 * Math.PI / (sampleCount + 1);
            double[] w = new double[sampleCount];
            for (int i = 0; i < w.Length; i++)
                w[i] = 0.5 - 0.5 * Math.Cos(k * (i + 1));
            return w;
        }

        void MinMax(double[] a, int w, double[] minval, double[] maxval)
        {
            LinkedList<int> u = new LinkedList<int>();
            LinkedList<int> l = new LinkedList<int>();
            for (int i = 1; i < a.Length; ++i)
            {
                if (i >= w)
                {
                    maxval[i - w] = a[u.Count > 0 ? u.First.Value : i - 1];
                    minval[i - w] = a[l.Count > 0 ? l.First.Value : i - 1];
                }
                if (a[i] > a[i - 1])
                {
                    l.AddLast(i - 1);
                    if (i == w + l.First.Value)
                        l.RemoveFirst();
                    while (u.Count > 0)
                    {
                        if (a[i] <= a[u.Last.Value])
                        {
                            if (i == w + u.First.Value) u.RemoveFirst();
                            break;
                        }
                        u.RemoveLast();
                    }
                }
                else
                {
                    u.AddLast(i - 1);
                    if (i == w + u.First.Value) u.RemoveFirst();
                    while (l.Count > 0)
                    {
                        if (a[i] >= a[l.Last.Value])
                        {
                            if (i == w + l.First.Value) l.RemoveFirst();
                            break;
                        }
                        l.RemoveLast();
                    }
                }
            }
        }

        double[] Abs(double[] signal)
        {
            double[] result = new double[signal.Length];
            for (int i = 0; i < signal.Length; i++)
                result[i] = Math.Abs(signal[i]);
            return result;
        }

        double[] Sub(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                throw new Exception();
            double[] r = new double[a.Length];
            for (int i = 0; i < r.Length; i++)
                r[i] = a[i] - b[i];
            return r;
        }

        double[] Mul(double[] signal, double value)
        {
            double[] r = new double[signal.Length];
            for (int i = 0; i < signal.Length; i++)
                r[i] = signal[i] * value;
            return r;
        }

        double[] Mul(double[] a, double[] b)
        {
            if (a.Length != b.Length)
                throw new Exception();
            double[] r = new double[a.Length];
            for (int i = 0; i < r.Length; i++)
                r[i] = a[i] * b[i];
            return r;
        }

        double[] Integrate(double[] src, int sampleCount)
        {
            if (sampleCount < 2)
                return src;

            double[] result = new double[src.Length];
            double[] buf = new double[sampleCount];
            int bufPtr = 0;
            double sum = 0.0;

            for (int i = 0; i < src.Length; i++)
            {
                sum -= buf[bufPtr];
                buf[bufPtr] = src[i];
                sum += src[i];
                bufPtr = (bufPtr + 1) % sampleCount;
                result[i] = sum / sampleCount;
            }
            return result;
        }

        double[] Normalize(double[] signal)
        {
            if (signal.Length == 0)
                return signal;

            double maxAbs = MaxAbs(signal);
            double[] normSignal = new double[signal.Length];
            for (int i = 0; i < signal.Length; i++)
                normSignal[i] = signal[i] / maxAbs;
            return normSignal;
        }

        double MaxAbs(double[] signal)
        {
            if (signal.Length == 0)
                throw new Exception();

            double min = signal[0];
            double max = signal[0];
            for (int i = 1; i < signal.Length; i++)
            {
                double val = signal[i];
                if (val < min)
                    min = val;
                else if (val > max)
                    max = val;
            }

            return Math.Max(Math.Abs(min), Math.Abs(max));
        }

        Complex[,] Mul(Complex[,] a, Complex[,] b)
        {
            int n = a.GetLength(0);
            int m = a.GetLength(1);
            int p = b.GetLength(1);
            if (b.GetLength(0) != m)
                throw new Exception();
            Complex[,] r = new Complex[n, p];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < p; j++)
                    for (int k = 0; k < m; k++)
                        r[i, j] += a[i, k] * b[k, j];
            return r;
        }

        Complex[,] Identity(int size)
        {
            Complex[,] r = new Complex[size, size];
            for (int i = 0; i < size; i++)
                r[i, i] = Complex.One;
            return r;
        }

        Complex[,] Inverse(Complex[,] m)
        {
            int n = m.GetLength(0);
            if (m.GetLength(1) != n)
                throw new ArgumentException("Matrix must be square");

            Complex[,] a = (Complex[,])m.Clone();
            Complex[,] b = Identity(n);

            for (int p = 0; p < n; p++)
            {
                int maxI = p;
                double maxAI = Complex.Abs(a[p, p]);
                for (int i = p + 1; i < n; i++)
                {
                    double maxAICand = Complex.Abs(a[i, p]);
                    if (maxAICand > maxAI)
                    {
                        maxI = i;
                        maxAI = maxAICand;
                    }
                }
                if (maxAI == 0.0)
                    return null;
                if (maxI != p)
                {
                    for (int j = 0; j < n; j++)
                    {
                        Complex temp = a[p, j];
                        a[p, j] = a[maxI, j];
                        a[maxI, j] = temp;
                        temp = b[p, j];
                        b[p, j] = b[maxI, j];
                        b[maxI, j] = temp;
                    }
                }

                Complex x = a[p, p];
                for (int j = 0; j < n; j++)
                {
                    a[p, j] /= x;
                    b[p, j] /= x;
                }

                for (int i = 0; i < n; i++)
                {
                    if (i == p)
                        continue;
                    x = a[i, p];
                    for (int j = 0; j < n; j++)
                    {
                        a[i, j] -= a[p, j] * x;
                        b[i, j] -= b[p, j] * x;
                    }
                }
            }

            return b;
        }

        Complex[] ToComplex(double[] a)
        {
            return a.Select(x => new Complex(x, 0.0)).ToArray();
        }

        Complex[] GetColumn(Complex[,] m, int index)
        {
            Complex[] r = new Complex[m.GetLength(0)];
            for (int i = 0; i < r.Length; i++)
                r[i] = m[i, index];
            return r;
        }

        Complex[,] Transpose(Complex[] row)
        {
            Complex[,] r = new Complex[row.Length, 1];
            for (int i = 0; i < row.Length; i++)
                r[i, 0] = row[i];
            return r;
        }

        Complex[,] ConjugateTranspose(Complex[,] m)
        {
            Complex[,] r = new Complex[m.GetLength(1), m.GetLength(0)];
            for (int i = 0; i < r.GetLength(0); i++)
                for (int j = 0; j < r.GetLength(1); j++)
                    r[i, j] = Complex.Conjugate(m[j, i]);
            return r;
        }

        Complex[,] ToeplitzMatrix(Complex[] c, Complex[] r)
        {
            if (c[0] != r[0])
                throw new Exception();
            Complex[,] x = new Complex[c.Length, r.Length];
            for (int i = 0; i < x.GetLength(0); i++)
            {
                for (int j = 0; j < x.GetLength(1); j++)
                {
                    int d = i - j;
                    x[i, j] = d > 0 ? c[d] : r[-d];
                }
            }
            return x;
        }

        double[] SignalStoD(short[] signal)
        {
            double[] signalD = new double[signal.Length];
            for (int i = 0; i < signal.Length; i++)
                signalD[i] = signal[i] / 32768.0;
            return signalD;
        }

        short[] SignalDtoS(double[] signal, bool dither = false)
        {
            Random rnd = new Random(12345);
            short[] signalS = new short[signal.Length];
            if (dither)
            {
                for (int i = 0; i < signal.Length; i++)
                {
                    double r = (rnd.NextDouble() + rnd.NextDouble()) - 1;
                    signalS[i] = Convert.ToInt16(signal[i] * 32766.0 + r);
                }
            }
            else
            {
                for (int i = 0; i < signal.Length; i++)
                    signalS[i] = Convert.ToInt16(signal[i] * 32767.0);
            }
            return signalS;
        }

        public static void SaveWav(string path, short[] signal, int sampleRate = sampleRate)
        {
            FileStream fs = File.Open(path, FileMode.Create);
            fs.Write(new byte[] { 0x52, 0x49, 0x46, 0x46 }, 0, 4); // "RIFF"
            fs.Write(BitConverter.GetBytes((uint)(36 + signal.Length * 2)), 0, 4);
            fs.Write(new byte[] { 0x57, 0x41, 0x56, 0x45 }, 0, 4); // "WAVE"
            fs.Write(new byte[] { 0x66, 0x6D, 0x74, 0x20 }, 0, 4); // "fmt"
            fs.Write(BitConverter.GetBytes((uint)(16)), 0, 4);
            fs.Write(BitConverter.GetBytes((ushort)(1)), 0, 2);
            fs.Write(BitConverter.GetBytes((ushort)(1)), 0, 2); // mono
            fs.Write(BitConverter.GetBytes((uint)(sampleRate)), 0, 4); // Hz
            fs.Write(BitConverter.GetBytes((uint)(sampleRate * 2)), 0, 4);
            fs.Write(BitConverter.GetBytes((ushort)(2)), 0, 2);
            fs.Write(BitConverter.GetBytes((ushort)(16)), 0, 2); // bps
            fs.Write(new byte[] { 0x64, 0x61, 0x74, 0x61 }, 0, 4); // "data"
            fs.Write(BitConverter.GetBytes((uint)(signal.Length * 2)), 0, 4);
            foreach (short v in signal)
                fs.Write(BitConverter.GetBytes(v), 0, 2);
            fs.Close();
        }
    }
}
