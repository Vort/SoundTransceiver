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

    class BitInfo : IComparable<BitInfo>
    {
        static public implicit operator double(BitInfo bit)
        {
            return bit.Value;
        }

        public int CompareTo(BitInfo other)
        {
            return Value.CompareTo(other.Value);
        }

        public double Value;
        public int Offset;
    }

    class Detector
    {
        public const int sampleRate = 44100;

        public double bitrate = 100;
        public double carrierFreq = 1000;

        double maxFreqDeviation = 0.005;

        int recvRrcBitCount = 4;
        double rrcBeta = 0.8;


        Diagram bitLevelsDiagram;


        public Detector(Diagram bitLevelsDiagram)
        {
            this.bitLevelsDiagram = bitLevelsDiagram;
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

        public byte[] Detect(short[] signal, out double snr)
        {
            if (signal.Length == 0)
                throw new SignalException("No data");

            bool debug = false;

            double[] D1 = SignalStoD(signal);
            if (debug) SaveWav("d1.wav", SignalDtoS(Normalize(D1)));

            double bandwidth = (1 + rrcBeta) * bitrate;
            double freq1 = (carrierFreq - bandwidth / 2.0) * (1.0 - maxFreqDeviation);
            double freq2 = (carrierFreq + bandwidth / 2.0) * (1.0 + maxFreqDeviation);
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
                throw new SignalException("Signal not found");
            if (signalEnd == 0)
                throw new SignalException("Signal not found");

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
                throw new SignalException("Incorrect signal");

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

            freq1 = (carrierFreq - carrierFreq * maxFreqDeviation) * 2;
            freq2 = (carrierFreq + carrierFreq * maxFreqDeviation) * 2;
            double estFreq = EstimateFrequency(D2s, freq1, freq2) / 2;

            // estFreq = freq * 0.99833;

            // freqDeviation = Math.Abs((estFreq - carrierFreq) / carrierFreq) * 1e6;

            double estBr = bitrate * estFreq / carrierFreq;

            double[] rrc = MakeRootRaisedCosine(
                sampleRate / estBr, recvRrcBitCount, rrcBeta);

            double[] D35I = new double[signalEnd2 - signalStart2 + rrc.Length];
            double[] D35Q = new double[D35I.Length];

            for (int i = 0; i < D35I.Length; i++)
            {
                double signalSample = D1[signalStart2 + i - rrc.Length / 2];
                double phase = 2 * Math.PI * estFreq * i / sampleRate;
                D35I[i] = signalSample * Math.Cos(phase);
                D35Q[i] = -signalSample * Math.Sin(phase);
            }

            //if (debug) SaveWav("d35i.wav", SignalDtoS(Normalize(D35I)));


            var D35If = Convolution(rrc, D35I);
            var D35Qf = Convolution(rrc, D35Q);


            var ph = GetPskPhase(D35If, D35Qf);

            double ca = Math.Cos(ph);
            double sa = Math.Sin(ph);
            for (int t = 0; t < D35If.Length; t++)
            {
                var i2 = D35If[t] * ca - D35Qf[t] * sa;
                var q2 = D35If[t] * sa + D35Qf[t] * ca;
                D35If[t] = i2;
                D35Qf[t] = q2;
            }

            double maxAbsIf = MaxAbs(D35If);
            double maxAbsQf = MaxAbs(D35Qf);
            double maxAbsIfQf = Math.Max(maxAbsIf, maxAbsQf);

            if (debug) SaveWav("d35if.wav", SignalDtoS(Mul(D35If, 1.0 / maxAbsIfQf)));
            if (debug) SaveWav("d35qf.wav", SignalDtoS(Mul(D35Qf, 1.0 / maxAbsIfQf)));

            var vecX = new List<double>();
            var vecY = new List<double>();
            for (int i = 1; i < D35If.Length; i++)
            {
                var s1 = D35If[i - 1];
                var s2 = D35If[i];
                if (s1 < 0 != s2 < 0)
                {
                    var adj = s1 / (s1 - s2);
                    var angle = 2 * Math.PI * ((i - 1 + adj) * estBr / sampleRate % 1.0);
                    vecX.Add(Math.Cos(angle));
                    vecY.Add(Math.Sin(angle));
                }
            }


            bool firstBit = true;
            bool inversion = false;
            var bits = new List<bool>();
            var bitLevels = new List<double>();
            double[] D4 = new double[D35If.Length];
            double tf = (Math.Atan2(vecY.Sum(), vecX.Sum()) + Math.PI) / (2 * Math.PI) * sampleRate / estBr;
            for (;;)
            {
                var ti = (int)Math.Round(tf);
                if (ti >= D35If.Length)
                    break;
                bool bit = D35If[ti] > 0.0;
                if (firstBit)
                {
                    inversion = bit;
                    firstBit = false;
                }
                bit ^= inversion;
                bits.Add(bit);
                double bitLevel = Math.Abs(D35If[ti]) / maxAbsIfQf;
                D4[ti] = (bit ? 1.0 : -1.0) * bitLevel;
                bitLevels.Add(bitLevel);

                tf += sampleRate / estBr;
            }

            if (debug) SaveWav("d4.wav", SignalDtoS(D4));

            // Последние биты иногда содержат шум
            bitLevels.RemoveRange(bitLevels.Count - 3, 3);
            /*
            bitLevelMin = (int)(bitLevels.Min() * 100.0);
            bitLevelAvg = (int)(bitLevels.Average() * 100.0);
            bitLevelMax = (int)(bitLevels.Max() * 100.0);
            */


            BitContainer bc = new BitContainer();

            int stage = 0;
            for (int i = 0; i < bits.Count; i++)
            {
                bool bit = bits[i];
                if (stage == 0)
                {
                    if (bit)
                        break;
                    else
                        stage = 1;
                }
                else if (stage == 1)
                {
                    if (bit)
                        stage = 2;
                }
                else if (stage == 2)
                {
                    if (!bit)
                        stage = 3;
                    else
                        break;
                }
                else if (stage == 3)
                {
                    bc.Add(bit);
                }
            }

            bitLevelsDiagram.Fill(bitLevels.ToArray(), 0.001, 1.0, 40);

            var data = bc.ExtractData();
            if (debug) File.WriteAllBytes("bytes", data);
            return data;
        }

        double GetPskPhase(double[] i, double[] q)
        {
            double[] rotI = new double[i.Length];
            double[] rotQ = new double[i.Length];

            double sumX = 0.0;
            double sumY = 0.0;
            for (int t = 0; t < i.Length; t++)
            {
                var a = Math.Atan2(q[t], i[t]) * 2.0;
                var r = Math.Sqrt(i[t] * i[t] + q[t] * q[t]);
                var px = r * Math.Cos(a);
                var py = r * Math.Sin(a);
                rotI[t] = px;
                rotQ[t] = py;
                sumX += px;
                sumY += py;
            }

            return Math.PI - Math.Atan2(sumY, sumX) / 2.0;
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

        double[] MakeRootRaisedCosine(double bitDuration, int bitCount, double beta)
        {
            int signalLen = (int)(bitCount * bitDuration) | 1;
            double[] rootRaisedCosine = new double[signalLen];
            for (int i = 0; i < signalLen; i++)
                rootRaisedCosine[i] = RootRaisedCosine(i - signalLen / 2, bitDuration, beta);
            return rootRaisedCosine;
        }

        double[] PadLeft(int sampleCount, double[] signal)
        {
            return new double[sampleCount].Concat(signal).ToArray();
        }

        double[] ApplyGaussianWindow(double[] Signal)
        {
            return AModulation(Signal, MakeGaussian(Signal.Length, 1));
        }

        double[] AModulation(double[] DataSignal, double[] AmpSignal)
        {
            double[] Signal = new double[Math.Min(DataSignal.Length, AmpSignal.Length)];

            for (int i = 0; i < Signal.Length; i++)
                Signal[i] = DataSignal[i] * AmpSignal[i];

            return Signal;
        }

        double[] MakeGaussian(int SampleCount, double K)
        {
            double A = Math.PI * K;
            double[] Gaussian = new double[SampleCount];

            for (int i = 0; i < SampleCount; i++)
            {
                double X = (2 * i / (double)SampleCount - 1);
                Gaussian[i] = Math.Exp(-A * X * X);
            }
            return Gaussian;
        }

        double[] MakeBandPassFilter(double Freq1, double Freq2, int SampleCount)
        {
            if (Freq1 <= 0)
                return MakeLowPassFilter(Freq2, SampleCount);
            else
            {
                double[] D1 = MakeLowPassFilter(Freq2, SampleCount);
                double[] D2 = MakeLowPassFilter(Freq1, SampleCount);
                double[] D3 = Sub(D1, D2);
                return D3;
            }
        }

        double[] MakeLowPassFilter(double CutoffFreq, int SampleCount)
        {
            double W = 2 * Math.PI * (CutoffFreq / sampleRate);
            double A = W / Math.PI;

            int N = SampleCount / 2;

            double[] IR = new double[SampleCount];

            for (int i = 0; i < N + 1; i++)
            {
                if (i == 0)
                    IR[N] = A;
                else
                {
                    double V = A * (Math.Sin(i * W) / (i * W));
                    IR[N + i] = V;
                    IR[N - i] = V;
                }
            }
            return IR;
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

        double[] Abs(double[] W)
        {
            double[] R = new double[W.Length];
            for (int i = 0; i < W.Length; i++)
                R[i] = Math.Abs(W[i]);
            return R;
        }

        double[] Sub(double[] W1, double[] W2)
        {
            int L = Math.Min(W1.Length, W2.Length);
            double[] R = new double[L];

            for (int i = 0; i < L; i++)
                R[i] = W1[i] - W2[i];

            return R;
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

        double[] Convolution(double[] W1, double[] W2)
        {
            if (W2.Length <= W1.Length)
                return new double[0];

            double[] R = new double[W2.Length - W1.Length];

            int blockSize = 512;
            int blockCount = (R.Length + blockSize - 1) / blockSize;

            Parallel.For(0, blockCount, i =>
                {
                    int startSample = i * blockSize;
                    int xendSample = blockSize;
                    if (i == blockCount - 1)
                        xendSample = R.Length - blockSize * i;
                    xendSample += startSample;
                    for (int j = startSample; j < xendSample; j++)
                    {
                        double Sum = 0;
                        for (int k = 0; k < W1.Length; k++)
                            Sum += W1[k] * W2[j + k];
                        R[j] = Sum;
                    }
                });
            return R;
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

        double[] Mul(double[] signal, double value)
        {
            double[] r = new double[signal.Length];

            for (int i = 0; i < signal.Length; i++)
                r[i] = signal[i] * value;

            return r;
        }


        double[] SignalStoD(short[] Signal)
        {
            double[] SignalD = new double[Signal.Length];
            for (int i = 0; i < Signal.Length; i++)
                SignalD[i] = Signal[i] / 32768.0;
            return SignalD;
        }

        short[] SignalDtoS(double[] Signal)
        {
            short[] SignalS = new short[Signal.Length];
            for (int i = 0; i < Signal.Length; i++)
                SignalS[i] = (short)(Signal[i] * 32767.0);
            return SignalS;
        }



        void SaveWav(string Path, short[] Signal)
        {
            FileStream FS = File.Open(Path, FileMode.Create);
            FS.Write(new byte[] { 0x52, 0x49, 0x46, 0x46 }, 0, 4); // "RIFF"
            FS.Write(BitConverter.GetBytes((uint)(36 + Signal.Length * 2)), 0, 4);
            FS.Write(new byte[] { 0x57, 0x41, 0x56, 0x45 }, 0, 4); // "WAVE"
            FS.Write(new byte[] { 0x66, 0x6D, 0x74, 0x20 }, 0, 4); // "fmt"
            FS.Write(BitConverter.GetBytes((uint)(16)), 0, 4);
            FS.Write(BitConverter.GetBytes((ushort)(1)), 0, 2);
            FS.Write(BitConverter.GetBytes((ushort)(1)), 0, 2); // mono
            FS.Write(BitConverter.GetBytes((uint)(44100)), 0, 4); // Hz
            FS.Write(BitConverter.GetBytes((uint)(44100 * 2)), 0, 4);
            FS.Write(BitConverter.GetBytes((ushort)(2)), 0, 2);
            FS.Write(BitConverter.GetBytes((ushort)(16)), 0, 2); // bps
            FS.Write(new byte[] { 0x64, 0x61, 0x74, 0x61 }, 0, 4); // "data"
            FS.Write(BitConverter.GetBytes((uint)(Signal.Length * 2)), 0, 4);
            foreach (short V in Signal)
                FS.Write(BitConverter.GetBytes(V), 0, 2);

            FS.Close();
        }

        double[] Normalize2(double[] signal)
        {
            if (signal.Length == 0)
                return signal;

            double min = signal.Min();
            double max = signal.Max();
            double delta = max - min;
            double[] normSignal = new double[signal.Length];
            for (int i = 0; i < signal.Length; i++)
                normSignal[i] = ((signal[i] - min) / delta) * 2 - 1;
            return normSignal;
        }

        double[] Normalize(double[] Signal)
        {
            if (Signal.Length == 0)
                return Signal;

            double Min = Signal.Min();
            double Max = Signal.Max();
            double MaxAbs = Math.Max(Math.Abs(Min), Math.Abs(Max));

            double[] NormSignal = new double[Signal.Length];
            for (int i = 0; i < Signal.Length; i++)
                NormSignal[i] = (Signal[i] / MaxAbs);
            return NormSignal;
        }

        BitInfo[] Normalize(BitInfo[] bits)
        {
            if (bits.Length == 0)
                return bits;

            double min = bits.Min();
            double max = bits.Max();
            double maxAbs = Math.Max(Math.Abs(min), Math.Abs(max));

            BitInfo[] normalized = new BitInfo[bits.Length];
            for (int i = 0; i < bits.Length; i++)
                normalized[i] = new BitInfo { Offset = bits[i].Offset, Value = bits[i].Value / maxAbs };
            return normalized;
        }
    }
}
