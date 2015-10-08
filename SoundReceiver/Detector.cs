using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

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
        public double bitrate = 100;
        public double sineFreq1 = 800;
        public double sineFreq2 = 1200;

        public const int m_SampleRate = 44100;

        Diagram m_DG1;
        Diagram m_DG2;

        public Detector(Diagram DG1, Diagram DG2)
        {
            m_DG1 = DG1;
            m_DG2 = DG2;
        }

        public byte[] Detect(short[] signal, ref string snr)
        {
            if (signal.Length == 0)
                throw new SignalException("No data");

            bool debug = false;
            double sineFreqC = (sineFreq1 + sineFreq2) / 2;

            double[] D1 = SignalStoD(signal);
            if (debug) SaveWav("e:\\_Research\\Sound\\RawSignal_D1.wav", SignalDtoS(Normalize(D1)));

            int F1size = 127;
            double[] F1 = ApplyGaussianWindow(MakeBandPassFilter(
                sineFreq1 - bitrate * 0.5,
                sineFreq2 + bitrate * 0.5, F1size));
            double[] D2 = Convolution(F1, D1);
            if (debug) SaveWav("e:\\_Research\\Sound\\RawSignal_D2.wav", SignalDtoS(Normalize(D2)));

            double[] D21 = Abs(D2);
            //if (debug) SaveWav("e:\\_Research\\Sound\\RawSignal_D21.wav", SignalDtoS(Normalize(D21)));

            int bitLen = (int)(m_SampleRate / bitrate);

            double[] D22 = Integrate(D21, bitLen);
            if (debug) SaveWav("e:\\_Research\\Sound\\RawSignal_D22.wav", SignalDtoS(Normalize(D22)));
            double[] D23 = Integrate(D21, bitLen * 8);
            //if (debug) SaveWav("e:\\_Research\\Sound\\RawSignal_D23.wav", SignalDtoS(Normalize(D23)));

            double[] D24 = new double[D22.Length];
            double[] D25 = new double[D22.Length];
            MinMax(D22, bitLen * 8, D24, D25);
            //if (debug) SaveWav("e:\\_Research\\Sound\\RawSignal_D24.wav", SignalDtoS(Normalize(D24)));
            //if (debug) SaveWav("e:\\_Research\\Sound\\RawSignal_D25.wav", SignalDtoS(Normalize(D25)));

            double noiseLevel = 0.0;
            int signalStart = 0;
            int signalEnd = 0;
            for (int i = bitLen * 8; i < D23.Length; i++)
            {
                if ((signalStart == 0) &&
                    (D24[i] > 2.0 * D23[i]))
                {
                    noiseLevel = D23[i];
                    signalStart = i;
                }
                if ((signalStart != 0) &&
                    (signalEnd == 0) &&
                    (D25[i] < 2.0 * noiseLevel))
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
            double signalStr = (D24[signalCenter] + D25[signalCenter]) / 2 - noiseLevel;
            snr = ((int)(Math.Log10(signalStr / noiseLevel) * 10)).ToString() + " dB";

            int signalStart2 = 0;
            int signalEnd2 = 0;
            double signalAvgMin = D24[signalCenter];
            double signalAvgMax = D25[signalCenter];
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
                throw new SignalException("Signal not found");

            signalStart2 += bitLen / 2;
            signalEnd2 -= bitLen / 2;

            if (debug)
            {
                D22 = Normalize(D22);
                D22[signalStart] = -1.0;
                D22[signalEnd] = -1.0;
                D22[signalStart2] = 1.0;
                D22[signalEnd2] = 1.0;
                SaveWav("e:\\_Research\\Sound\\RawSignal_D22.wav", SignalDtoS((D22)));
            }

            int F2size = 511;
            int F3size = 511;
            double[] F2 = ApplyGaussianWindow(MakeLowPassFilter(
                Math.Abs(sineFreq2 - sineFreq1) / 2 + bitrate, F2size));
            double[] F3 = ApplyGaussianWindow(MakeLowPassFilter(bitrate, F3size));

            signalStart2 += F1size / 2;
            signalEnd2 += F1size / 2;

            signalStart2 -= bitLen / 2;
            signalEnd2 -= bitLen / 2;

            double[] D35I = new double[signalEnd2 - signalStart2 + F3size + F2size];
            double[] D35Q = new double[signalEnd2 - signalStart2 + F3size + F2size];
            for (int i = 0; i < D35I.Length; i++)
            {
                D35I[i] = D1[signalStart2 - (F3size + F2size) / 2 + i] * Math.Cos(2 * Math.PI * sineFreqC * (i / (double)m_SampleRate));
                D35Q[i] = -D1[signalStart2 - (F3size + F2size) / 2 + i] * Math.Sin(2 * Math.PI * sineFreqC * (i / (double)m_SampleRate));
            }

            double[] D35If = Convolution(F2, D35I);
            double[] D35Qf = Convolution(F2, D35Q);
            double[] D36 = new double[D35If.Length];
            double[] D37 = new double[D35If.Length];
            for (int i = 0; i < D35If.Length; i++)
            {
                D36[i] = Math.Atan2(D35Qf[i], D35If[i]);
                if (i != 0)
                {
                    double delta = D36[i] - D36[i - 1];
                    if (delta > Math.PI)
                        delta -= Math.PI * 2;
                    if (delta < -Math.PI)
                        delta += Math.PI * 2;
                    D37[i] = delta / Math.PI;
                }
            }

            double[] D38 = Normalize2(Convolution(F3, D37));

            signalStart2 -= (F3size + F2size) / 2;
            signalEnd2 -= (F3size + F2size) / 2;

            double[] D5 = new double[D38.Length];
            Array.Copy(D38, D5, D5.Length);

            // Детектор фронтов
            int pf1 = 0;
            int pf2 = 0;
            int pf3 = 0;
            int pr1 = 0;
            int pr2 = 0;
            int pr3 = 0;
            int ff = 0;
            int fr = 0;
            List<int> fronts = new List<int>();
            for (int i = 1; i < D5.Length; i++)
            {
                if ((D5[i - 1] > 0.3) && (D5[i] <= 0.3))
                    pf1 = i;
                if ((D5[i - 1] < 0.3) && (D5[i] >= 0.3))
                    pr3 = i;
                if ((D5[i - 1] > 0.0) && (D5[i] <= 0.0))
                    pf2 = i;
                if ((D5[i - 1] < 0.0) && (D5[i] >= 0.0))
                    pr2 = i;
                if ((D5[i - 1] > -0.3) && (D5[i] <= -0.3))
                    pf3 = i;
                if ((D5[i - 1] < -0.3) && (D5[i] >= -0.3))
                    pr1 = i;
                if ((pf1 != 0) && (pf2 > pf1) && (pf3 > pf2))
                {
                    ff = pf2;
                    D5[ff] = -1.0;
                    pf1 = 0;
                    pf2 = 0;
                    pf3 = 0;
                }
                if ((pr1 != 0) && (pr2 > pr1) && (pr3 > pr2))
                {
                    fr = pr2;
                    D5[fr] = -1.0;
                    pr1 = 0;
                    pr2 = 0;
                    pr3 = 0;
                }

                if ((ff != 0) && (fr > ff))
                {
                    int favg = fr - (((fr - ff) + bitLen / 2) % bitLen - bitLen / 2) / 2;
                    fronts.Add(favg);
                    D5[favg] = 1.0;
                    fr = 0;
                    ff = 0;
                }
            }

            if (debug)
                SaveWav("e:\\_Research\\Sound\\RawSignal_D5.wav", SignalDtoS(PadLeft(signalStart2, D5)));

            if (fronts.Count == 0)
                throw new SignalException("Incorrect signal");

            int startingBitCount = (fronts[0]) / bitLen;
            int endingBitCount = (signalEnd2 - signalStart2 - fronts[fronts.Count - 1]) / bitLen;

            if (startingBitCount < 1)
                throw new SignalException("Incorrect signal");
            if (endingBitCount < 0)
                throw new SignalException("Incorrect signal");

            List<BitInfo> rawBits = new List<BitInfo>();
            rawBits.AddRange(GetRawBits2(D38,
                fronts[0] - bitLen * startingBitCount, startingBitCount, bitrate));
            for (int i = 1; i < fronts.Count; i++)
            {
                rawBits.AddRange(GetRawBits2(D38,
                    fronts[i - 1], Convert.ToInt32(
                    (fronts[i] - fronts[i - 1]) / (double)bitLen), bitrate));
            }
            rawBits.AddRange(GetRawBits2(D38,
                fronts[fronts.Count - 1], endingBitCount, bitrate));

            rawBits = new List<BitInfo>(Normalize(rawBits.ToArray()));


            int stage = 0;
            List<BitInfo> fltRawBits = new List<BitInfo>();
            for (int i = 0; i < rawBits.Count; i++)
            {
                bool bit = rawBits[i].Value > 0;
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
                    fltRawBits.Add(rawBits[i]);
                }
            }

            if (stage != 3)
                throw new SignalException("Incorrect signal");

            double[] bitsValue = new double[fltRawBits.Count];
            for (int i = 0; i < fltRawBits.Count; i++)
                bitsValue[i] = fltRawBits[i].Value;

            m_DG1.Fill(bitsValue, -1.0, -0.1, 18);
            m_DG2.Fill(bitsValue, 0.1, 1.0, 18);

            BitContainer bc = new BitContainer();
            for (int i = 0; i < fltRawBits.Count; i++)
                bc.Add(fltRawBits[i].Value > 0);

            return bc.ExtractData();
        }

        BitInfo[] GetRawBits2(double[] signal, int startOffset, int bitCount, double bitrate)
        {
            int bitLen = (int)(m_SampleRate / bitrate);
            BitInfo[] rawBits = new BitInfo[bitCount];

            for (int i = 0; i < bitCount; i++)
            {
                double rawBit = 0.0;
                int bitOffset = startOffset + i * bitLen;
                for (int j = 0; j < bitLen / 2; j++)
                    rawBit += signal[bitOffset + j + bitLen / 4];
                rawBit /= bitLen / 2;
                rawBits[i] = new BitInfo { Value = rawBit, Offset = bitOffset };
            }
            return rawBits;
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
            double W = 2 * Math.PI * (CutoffFreq / m_SampleRate);
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

            for (int i = 0; i < R.Length; i++)
            {
                double Sum = 0;
                for (int j = 0; j < W1.Length; j++)
                    Sum += W1[j] * W2[i + j];
                R[i] = Sum;
            }
            return R;
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
