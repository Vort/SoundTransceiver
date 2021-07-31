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
        public const int sampleRate = 44100;

        public double bitrate = 100;
        public double carrierFreq = 1000;

        double maxFreqDeviation = 0.005;

        sbyte[] equalizerTrainSequence = new sbyte[]
        {
            1, 1, 1, 1, 1, -1, 1, -1,
            -1, -1, 1, -1, 1, 1, 1, 1,
            1, -1, 1, -1, -1, 1, 1, -1,
            -1, -1, -1, 1, -1, -1, -1, -1
        };
        int sendRrcBitCount = 64;
        int recvRrcBitCount = 4;
        double rrcBeta = 0.8;

        int payloadSizeLsbSize = 3;

        Diagram bitLevelsDiagram;


        public Detector(Diagram bitLevelsDiagram)
        {
            this.bitLevelsDiagram = bitLevelsDiagram;
        }

        public byte[] Detect(short[] signal, out double snr)
        {
            if (signal.Length == 0)
                throw new SignalException("No data");

            bool debug = false;

            SaveWav($"r_{carrierFreq}_{bitrate}.wav", signal);

            double[] D1 = SignalStoD(signal);

            double bandwidth = (1 + rrcBeta) * bitrate;
            double freq = carrierFreq;
            double freq1 = (freq - bandwidth / 2.0) * (1.0 - maxFreqDeviation);
            double freq2 = (freq + bandwidth / 2.0) * (1.0 + maxFreqDeviation);
            int F1size = (int)(4.0 * sampleRate / (freq2 - freq1)) | 1;
            double[] F1 = ApplyGaussianWindow(MakeBandPassFilter(freq1, freq2, F1size));
            double[] D2 = Convolution(F1, D1);
            if (debug) SaveWav("d2.wav", SignalDtoS(Normalize(D2)));

            double[] D21 = Abs(D2);

            int bitLen = (int)(sampleRate / bitrate);

            double[] D22 = Integrate(D21, bitLen);
            if (debug) SaveWav("d22.wav", SignalDtoS(Normalize(D22)));
            double[] D23 = Integrate(D21, bitLen * 8);
            //if (debug) SaveWav("d23.wav", SignalDtoS(D23));

            double[] D24 = new double[D22.Length];
            double[] D25 = new double[D22.Length];
            MinMax(D22, bitLen * 8, D24, D25);

            //if (debug) SaveWav("d24.wav", SignalDtoS(D24));
            //if (debug) SaveWav("d25.wav", SignalDtoS(D25));


            double noiseLevel = 0.0;
            int signalStart = 0;
            int signalEnd = 0;
            for (int i = bitLen * 8; i < D23.Length; i++)
            {
                if ((signalStart == 0) &&
                    (D24[i] > 2.0 * D23[i]))
                {
                    noiseLevel = D23[i - bitLen];
                    signalStart = i;
                }
                if ((signalStart != 0) &&
                    (signalEnd == 0) &&
                    (D25[i] <= 2.0 * noiseLevel))
                {
                    signalEnd = i;
                    break;
                }
            }

            if (signalStart == 0)
                throw new SignalException("Signal start #1 is not found");
            if (signalEnd == 0)
                throw new SignalException("Signal end #1 is not found");

            int signalCenter = (signalStart + signalEnd) / 2;

            int signalStart2 = 0;
            int signalEnd2 = 0;
            double signalAvgMin = D24[signalCenter - bitLen * 4];
            for (int i = signalStart; i < signalEnd; i++)
            {
                if ((signalStart2 == 0) &&
                    (D24[i] > 0.5 * signalAvgMin))
                {
                    signalStart2 = i;
                }
                if ((signalStart2 != 0) &&
                    (signalEnd2 == 0) &&
                    (D25[i] < 0.5 * signalAvgMin))
                {
                    signalEnd2 = i;
                    break;
                }
            }
            if (signalEnd2 == 0)
                signalEnd2 = signalEnd;

            if (signalStart2 == 0)
                throw new SignalException("Signal start #2 is not found");

            double signalAvgLevel = D23[signalCenter];
            double signalStr = signalAvgLevel - noiseLevel;
            snr = 99;
            if (noiseLevel != 0.0)
                snr = Math.Round(Math.Log10(signalStr / noiseLevel) * 20);

            signalStart2 = signalStart2 + F1size / 2 - bitLen / 2 + bitLen / 4;
            signalEnd2 = signalEnd2 + F1size / 2 - bitLen / 2 - bitLen / 4;

            double[] D2s = new double[signalEnd2 - signalStart2];
            for (int i = 0; i < D2s.Length; i++)
            {
                double d2v = D2[i + signalStart2 - F1size / 2];
                D2s[i] = d2v * d2v;
            }

            if (debug) SaveWav("d2s.wav", SignalDtoS(Normalize(D2s)));

            freq1 = (freq - freq * maxFreqDeviation) * 2;
            freq2 = (freq + freq * maxFreqDeviation) * 2;
            double estFreq = EstimateFrequency(D2s, freq1, freq2) / 2;

            //freqDeviation = Math.Abs((estFreq - freq) / freq) * 1e6;


            double estBitrate = bitrate * estFreq / freq;
            int decFactor = (int)(sampleRate / estBitrate / 4);
            int decFilterSize = decFactor == 1 ? 0 : (int)(8 * sampleRate / estBitrate) | 1;
            double decSampleRate = (double)sampleRate / decFactor;
            double estBitlen = decSampleRate / estBitrate;



            double[] trainSignal = MakePulsesShape(
                equalizerTrainSequence, decSampleRate, estBitrate, rrcBeta, sendRrcBitCount);
            if (debug) SaveWav("trainSignal.wav", SignalDtoS(Mul(trainSignal, 0.67)));

            double[] rrc = Mul(MakeRootRaisedCosine(
                estBitlen, recvRrcBitCount, rrcBeta), 1 / estBitlen);

            if (debug) SaveWav("rrc.wav", SignalDtoS(rrc));



            int eqSize = (int)(8 * estBitlen) | 1;
            Complex[] D35 = new Complex[signalEnd2 - signalStart2 +
                (rrc.Length + eqSize + 2/*<--hack*/) * decFactor + decFilterSize];
            for (int i = 0; i < D35.Length; i++)
            {
                double signalSample = D1[signalStart2 + i - (rrc.Length + eqSize) * decFactor / 2 - decFilterSize / 2];
                double phase = 2 * Math.PI * estFreq * i / sampleRate;
                D35[i] = new Complex(
                    signalSample * Math.Cos(phase),
                    -signalSample * Math.Sin(phase));
            }

            if (decFactor != 1)
            {
                double[] flt = MakeLowPassFilter(sampleRate / 2.0 / decFactor, decFilterSize);
                D35 = Decimate(flt, D35, decFactor);
            }


            double[] D35I = D35.Select(x => x.Real).ToArray();
            double[] D35Q = D35.Select(x => x.Imaginary).ToArray();

            double maxAbsI = MaxAbs(D35I);
            double maxAbsQ = MaxAbs(D35Q);
            double maxAbsIQ = Math.Max(maxAbsI, maxAbsQ);

            if (debug) SaveWav("d35i.wav", SignalDtoS(Mul(D35I, 1.0 / maxAbsIQ)), (int)decSampleRate);
            if (debug) SaveWav("d35q.wav", SignalDtoS(Mul(D35Q, 1.0 / maxAbsIQ)), (int)decSampleRate);


            int trainOffset1 = Convert.ToInt32(estBitlen * 0.5 + eqSize / 2 + rrc.Length / 2);
            int trainSize1 = Convert.ToInt32(estBitlen * (equalizerTrainSequence.Length - 1));
            int trainSize2 = trainSize1 - eqSize + 1;
            int trainOffset3 = (int)(sendRrcBitCount * estBitlen) / 2;
            int trainOffset4 = trainOffset3 + Convert.ToInt32(eqSize / 2);
            int deltaRange = Convert.ToInt32(estBitlen / 2);

            Complex[,] r = ToeplitzMatrix(
                D35.Skip(trainOffset1 + eqSize - 1).Take(trainSize1 - eqSize + 1).ToArray(),
                D35.Skip(trainOffset1).Take(eqSize).Reverse().ToArray());
            Complex[,] s = ToeplitzMatrix(
                ToComplex(trainSignal.Skip(trainOffset4 + deltaRange).Take(trainSize2).ToArray()),
                ToComplex(trainSignal.Skip(trainOffset4 - deltaRange).Take(deltaRange * 2 + 1).Reverse().ToArray()));

            Complex[,] rct = ConjugateTranspose(r);
            Complex[,] rctrirct = Mul(Inverse(Mul(rct, r)), rct);

            Complex[,] f = Mul(rctrirct, s);
            Complex[,] p = Mul(Mul(ConjugateTranspose(s),
                Sub(Identity(s.GetLength(0)), Mul(r, rctrirct))), s);
            double[] adp = Abs(Diagonal(p));

            int minadpi = Array.IndexOf(adp, adp.Min());

            Complex[] eq = GetColumn(f, minadpi);

            // TODO: Make Convolution() work in classical way
            var D35f = Convolution(eq.Reverse().ToArray(), D35);

            var D35If = D35f.Select(x => x.Real).ToArray();
            var D35Qf = D35f.Select(x => x.Imaginary).ToArray();
            var D35If2 = Convolution(rrc, D35If);
            var D35Qf2 = Convolution(rrc, D35Qf);

            if (debug) SaveWav("d35if.wav", SignalDtoS(Mul(D35If, 0.25)), (int)decSampleRate);
            if (debug) SaveWav("d35qf.wav", SignalDtoS(Mul(D35Qf, 0.25)), (int)decSampleRate);

            if (debug) SaveWav("d35if2.wav", SignalDtoS(Mul(D35If2, 0.25)), (int)decSampleRate);
            if (debug) SaveWav("d35qf2.wav", SignalDtoS(Mul(D35Qf2, 0.25)), (int)decSampleRate);

            var bits = new List<bool>();
            var bitLevels = new List<double>();
            double[] D4 = new double[(D35If2.Length + 1/*<--hack*/) * decFactor];
            double tf = minadpi;
            for (;;)
            {
                int ti = (int)tf;
                double ts = tf - ti;

                if (ti >= D35If2.Length)
                    break;

                double[] rrcSubsample = MakeRootRaisedCosine(
                    estBitlen, recvRrcBitCount, rrcBeta, ts);

                double sample = Convolution(
                    rrcSubsample, D35If, ti) / estBitlen;

                bool bit = sample > 0.0;
                bits.Add(bit);

                double bitLevel = Math.Abs(sample);
                var ti2 = (int)Math.Round(tf * decFactor);
                D4[ti2] = (bit ? 1.0 : -1.0) * bitLevel;
                bitLevels.Add(bitLevel);

                tf += decSampleRate / estBitrate;
            }

            if (debug) SaveWav("d4.wav", SignalDtoS(Mul(D4, 0.25)));

            bool[] plszlsbb = bits.Skip(equalizerTrainSequence.Length).Take(payloadSizeLsbSize).ToArray();
            int plszlsb = 0;
            for (int i = 0; i < payloadSizeLsbSize; i++)
                plszlsb |= plszlsbb[i] ? (1 << payloadSizeLsbSize - i - 1) : 0;

            int plszEst = (bits.Count - equalizerTrainSequence.Length - payloadSizeLsbSize) / 8;
            int plsz = plszEst;
            // TODO: implement support for other payloadSizeLsbSize values
            int plszEstLsb = plszEst & 7;
            if (plszEstLsb != plszlsb)
            {
                plsz &= ~7;
                plsz |= plszlsb;
                if (plszEstLsb < plszlsb)
                {
                    plsz -= 8;
                    if (plsz < 0)
                        throw new SignalException("Wrong payload size");
                }
            }

            int skippedBitCount = bits.Count -
                equalizerTrainSequence.Length - payloadSizeLsbSize - plsz * 8;

            bits.RemoveRange(bitLevels.Count - skippedBitCount, skippedBitCount);
            bitLevels.RemoveRange(bitLevels.Count - skippedBitCount, skippedBitCount);
            //bitLevelMin = (int)(bitLevels.Min() * 100.0);

            bitLevelsDiagram.Fill(bitLevels.ToArray(), 0.001, 2.0, 39);

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
            double[] signalp2 = signal.Take(
                1 << (int)Math.Log(signal.Length, 2)).ToArray();

            var n = signalp2.Length;
            var signalp2c = DitFft2(PadLeft(n * 3, signalp2)).ToArray();

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

            var u = (64 * n) / (Math.Pow(Math.PI, 5) + 32 * Math.PI);
            var v = u * Math.Pow(Math.PI, 2) / 4;
            var wa = (vp1 - vm1) / (u * (vp1 + vm1) + v * vz);

            return ((double)maxPos / n + wa) * sampleRate / 4;
        }

        Complex[] Fourier(double[] signal, int size, int ofs, int stride)
        {
            var result = new Complex[size];
            double step = 2 * Math.PI / size;

            for (int k = 0; k < size; k++)
            {
                double si = 0.0;
                double sq = 0.0;
                double stepk = step * k;
                for (int n = 0; n < size; n++)
                {
                    double stepkn = stepk * n;
                    double sv = signal[stride * n + ofs];
                    si += sv * Math.Cos(stepkn);
                    sq -= sv * Math.Sin(stepkn);
                }
                result[k] = new Complex(si, sq);
            }
            return result;
        }

        Complex[] DitFft2(double[] signal, int size, int ofs, int s)
        {
            var result = new Complex[size];
            if (size <= 4)
                result = Fourier(signal, size, ofs, s);
            else
            {
                var step = -2.0 * Math.PI * Complex.ImaginaryOne / size;
                var p1 = DitFft2(signal, size / 2, ofs, s * 2);
                var p2 = DitFft2(signal, size / 2, ofs + s, s * 2);
                for (int k = 0; k < size / 2; k++)
                {
                    var e = Complex.Exp(step * k);
                    result[k] = p1[k] + e * p2[k];
                    result[k + size / 2] = p1[k] - e * p2[k];
                }
            }
            return result;
        }

        Complex[] DitFft2(double[] signal)
        {
            return DitFft2(signal, signal.Length, 0, 1);
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
            double epsilon = 1e-12;
            if (Math.Abs(t) < epsilon)
            {
                return 1.0 + beta * (4.0 / Math.PI - 1.0);
            }
            else if (Math.Abs(Math.Abs(t) - bitDuration / (4.0 * beta)) > epsilon)
            {
                double f = t / bitDuration;
                return (
                    Math.Sin((Math.PI * f) * (1.0 - beta)) +
                    (4.0 * beta * f) *
                    Math.Cos((Math.PI * f) * (1.0 + beta))) /
                    ((Math.PI * f) *
                    (1.0 - Math.Pow(4.0 * beta * f, 2.0)));
            }
            else
            {
                return (beta / Math.Sqrt(2.0)) *
                    ((1.0 + 2.0 / Math.PI) * Math.Sin(Math.PI / (4.0 * beta)) +
                    (1.0 - 2.0 / Math.PI) * Math.Cos(Math.PI / (4.0 * beta)));
            }
        }

        double[] MakeRootRaisedCosine(double bitDuration,
            int bitCount, double beta, double subsampleShift = 0.0)
        {
            int signalLen = (int)(bitCount * bitDuration) | 1;
            double[] rootRaisedCosine = new double[signalLen];
            for (int i = 0; i < signalLen; i++)
            {
                rootRaisedCosine[i] = RootRaisedCosine(
                    i - signalLen / 2 - subsampleShift, bitDuration, beta);
            }
            return rootRaisedCosine;
        }

        Complex[] Convolution(Complex[] kernel, Complex[] signal)
        {
            if (signal.Length <= kernel.Length)
                return new Complex[0];

            Complex[] r = new Complex[signal.Length - kernel.Length];

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
                    for (int k = 0; k < kernel.Length; k++)
                        sum += kernel[k] * signal[j + k];
                    r[j] = sum;
                }
            });
            return r;
        }

        double[] Convolution(double[] kernel, double[] signal)
        {
            if (signal.Length <= kernel.Length)
                return new double[0];

            double[] r = new double[signal.Length - kernel.Length];

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
                    for (int k = 0; k < kernel.Length; k++)
                        sum += kernel[k] * signal[j + k];
                    r[j] = sum;
                }
            });
            return r;
        }

        double Convolution(double[] kernel, double[] signal, int signalOffset)
        {
            double r = 0.0;
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

        double[] ApplyGaussianWindow(double[] signal)
        {
            return AModulation(signal, MakeGaussian(signal.Length, 1));
        }

        double[] AModulation(double[] dataSignal, double[] ampSignal)
        {
            double[] signal = new double[Math.Min(dataSignal.Length, ampSignal.Length)];

            for (int i = 0; i < signal.Length; i++)
                signal[i] = dataSignal[i] * ampSignal[i];

            return signal;
        }

        double[] MakeGaussian(int sampleCount, double k)
        {
            double a = Math.PI * k;
            double[] gaussian = new double[sampleCount];

            for (int i = 0; i < sampleCount; i++)
            {
                double x = (2 * i / (double)sampleCount - 1);
                gaussian[i] = Math.Exp(-a * x * x);
            }
            return gaussian;
        }

        double Lerp(double[] Signal, double X)
        {
            double F1 = Math.Floor(X);
            double F2 = F1 + 1;
            double F3 = F2 - X;
            double F4 = X - F1;

            int IF1 = (int)F1;
            int IF2 = IF1 + 1;

            double V1 = 0;
            double V2 = 0;
            if (IF1 >= 0)
                if (IF1 < Signal.Length)
                {
                    V1 = Signal[IF1];
                    if (F4 == 0)
                        return V1;
                }
            if (IF2 >= 0)
                if (IF2 < Signal.Length)
                    V2 = Signal[IF2];

            return V1 * F3 + V2 * F4;
        }

        void MinMax(double[] a, int w, double[] minval, double[] maxval)
        {
            LinkedList<int> U = new LinkedList<int>();
            LinkedList<int> L = new LinkedList<int>();
            for (int i = 1; i < a.Length; ++i)
            {
                if (i >= w)
                {
                    maxval[i - w] = a[U.Count > 0 ? U.First.Value : i - 1];
                    minval[i - w] = a[L.Count > 0 ? L.First.Value : i - 1];
                }
                if (a[i] > a[i - 1])
                {
                    L.AddLast(i - 1);
                    if (i == w + L.First.Value)
                        L.RemoveFirst();
                    while (U.Count > 0)
                    {
                        if (a[i] <= a[U.Last.Value])
                        {
                            if (i == w + U.First.Value) U.RemoveFirst();
                            break;
                        }
                        U.RemoveLast();
                    }
                }
                else
                {
                    U.AddLast(i - 1);
                    if (i == w + U.First.Value) U.RemoveFirst();
                    while (L.Count > 0)
                    {
                        if (a[i] >= a[L.Last.Value])
                        {
                            if (i == w + L.First.Value) L.RemoveFirst();
                            break;
                        }
                        L.RemoveLast();
                    }
                }
            }
            maxval[a.Length - w] = a[U.Count > 0 ? U.First.Value : a.Length - 1];
            minval[a.Length - w] = a[L.Count > 0 ? L.First.Value : a.Length - 1];
        }

        double[] Abs(double[] signal)
        {
            double[] result = new double[signal.Length];
            for (int i = 0; i < signal.Length; i++)
                result[i] = Math.Abs(signal[i]);
            return result;
        }

        double[] Sub(double[] W1, double[] W2)
        {
            int L = Math.Min(W1.Length, W2.Length);
            double[] R = new double[L];

            for (int i = 0; i < L; i++)
                R[i] = W1[i] - W2[i];

            return R;
        }

        double[] Mul(double[] signal, double value)
        {
            double[] r = new double[signal.Length];

            for (int i = 0; i < signal.Length; i++)
                r[i] = signal[i] * value;

            return r;
        }

        double[] Integrate(double[] Signal, double SampleCount)
        {
            double S = 0;
            double[] R = new double[Signal.Length];

            for (int i = 0; i < R.Length; i++)
            {
                S += Integrate(Signal, i, i + 1);
                S -= Integrate(Signal, i - SampleCount, i - SampleCount + 1);
                R[i] = S / SampleCount;
            }
            return R;
        }

        double Integrate(double[] Signal, double StartSample, double EndSample)
        {
            if (StartSample >= EndSample)
                return 0;

            double SF = Math.Floor(StartSample);
            double EF = Math.Floor(EndSample);

            if (SF == EF)
            {
                double V1 = Lerp(Signal, StartSample);
                double V2 = Lerp(Signal, EndSample);
                double V = (EndSample - StartSample) * (V2 + V1) / 2;
                return V;
            }
            else
            {
                double V1 = Lerp(Signal, StartSample);
                double V2 = Lerp(Signal, SF + 1);
                double V3 = (SF - StartSample + 1) * (V2 + V1) / 2;

                double V4 = Lerp(Signal, EF);
                double V5 = Lerp(Signal, EndSample);
                double V6 = (EndSample - EF) * (V4 + V5) / 2;

                double V = V3 + V6;

                int C = (int)(EF - SF - 1);
                for (int i = 0; i < C; i++)
                {
                    double VI1 = Lerp(Signal, SF + 1);
                    double VI2 = Lerp(Signal, SF + 2);
                    double VI3 = (VI1 + VI2) / 2;
                    V += VI3;
                }

                return V;
            }
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
            if (a.GetLength(1) != b.GetLength(0))
                throw new Exception();
            Complex[,] r = new Complex[a.GetLength(0), b.GetLength(1)];
            for (int i = 0; i < r.GetLength(0); i++)
                for (int j = 0; j < r.GetLength(1); j++)
                    for (int k = 0; k < a.GetLength(1); k++)
                        r[i, j] += a[i, k] * b[k, j];
            return r;
        }

        Complex[,] Sub(Complex[,] a, Complex[,] b)
        {
            if (a.GetLength(0) != b.GetLength(0) ||
                a.GetLength(1) != b.GetLength(1))
            {
                throw new Exception();
            }
            Complex[,] r = new Complex[a.GetLength(0), a.GetLength(1)];
            for (int i = 0; i < r.GetLength(0); i++)
                for (int j = 0; j < r.GetLength(1); j++)
                    r[i, j] = a[i, j] - b[i, j];
            return r;
        }

        Complex[,] Identity(int size)
        {
            Complex[,] r = new Complex[size, size];
            for (int i = 0; i < size; i++)
                r[i, i] = Complex.One;
            return r;
        }

        Complex[] Diagonal(Complex[,] m)
        {
            int n = m.GetLength(0);
            if (m.GetLength(1) != n)
                throw new ArgumentException("Matrix must be square");
            Complex[] r = new Complex[n];
            for (int i = 0; i < n; i++)
                r[i] = m[i, i];
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

        double[] Abs(Complex[] a)
        {
            double[] r = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
                r[i] = Complex.Abs(a[i]);
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

        void SaveWav(string path, short[] signal, int sampleRate = sampleRate)
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
