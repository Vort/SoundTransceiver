using System;
using System.Numerics;

namespace SoundReceiver
{
    class Fourier
    {
        Complex[][] twiddle;

        public Fourier()
        {
            twiddle = new Complex[17][];
            for (int i = 3; i < twiddle.Length; i++)
            {
                int sz = 1 << i;
                int hsz = 1 << i - 1;
                var step = -2.0 * Math.PI / sz;
                twiddle[i] = new Complex[hsz];
                for (int k = 0; k < hsz; k++)
                {
                    double stepk = step * k;
                    twiddle[i][k] = new Complex(
                        Math.Cos(stepk), Math.Sin(stepk));
                }
            }
        }

        public Complex[] Transform(double[] signal)
        {
            int sizePower = (int)Math.Log(signal.Length, 2);
            if (signal.Length != 1 << sizePower)
                throw new ArgumentException("Incorrect signal size");
            return DitFft2(signal, sizePower, 0, 1);
        }

        Complex[] DitFft2(double[] signal, int sizePower, int ofs, int s)
        {
            if (sizePower >= 4)
            {
                int size = 1 << sizePower;
                int halfSize = 1 << sizePower - 1;
                var p1 = DitFft2(signal, sizePower - 1, ofs, s * 2);
                var p2 = DitFft2(signal, sizePower - 1, ofs + s, s * 2);
                var result = new Complex[size];
                if (sizePower < twiddle.Length)
                {
                    for (int k = 0; k < halfSize; k++)
                    {
                        var ep2k = twiddle[sizePower][k] * p2[k];
                        result[k] = p1[k] + ep2k;
                        result[k + halfSize] = p1[k] - ep2k;
                    }
                }
                else
                {
                    var step = -2.0 * Math.PI / size;
                    for (int k = 0; k < halfSize; k++)
                    {
                        double stepk = step * k;
                        var ep2k = new Complex(
                            Math.Cos(stepk), Math.Sin(stepk)) * p2[k];
                        result[k] = p1[k] + ep2k;
                        result[k + halfSize] = p1[k] - ep2k;
                    }
                }
                return result;
            }
            else if (sizePower == 3)
                return Fourier8(signal, ofs, s);
            else if (sizePower == 2)
                return Fourier4(signal, ofs, s);
            else if (sizePower == 1)
                return Fourier2(signal, ofs, s);
            else
                return new Complex[1] { new Complex(signal[ofs], 0.0) };
        }

        Complex[] Fourier8(double[] signal, int ofs, int stride)
        {
            double sv0 = signal[ofs];
            double sv1 = signal[ofs + stride * 2];
            double sv2 = signal[ofs + stride * 4];
            double sv3 = signal[ofs + stride * 6];

            double sv4 = signal[ofs + stride];
            double sv5 = signal[ofs + stride * 3];
            double sv6 = signal[ofs + stride * 5];
            double sv7 = signal[ofs + stride * 7];

            double sv0psv2 = sv0 + sv2;
            double sv0msv2 = sv0 - sv2;
            double sv1psv3 = sv1 + sv3;

            double sv4psv6 = sv4 + sv6;
            double sv4msv6 = sv4 - sv6;
            double sv5psv7 = sv5 + sv7;
            double sv7msv5 = sv7 - sv5;

            var r0 = new Complex(sv0psv2 + sv1psv3 + sv4psv6 + sv5psv7, 0.0);
            var r2 = new Complex(sv0psv2 - sv1psv3, sv5psv7 - sv4psv6);
            var r4 = new Complex(sv0psv2 + sv1psv3 - sv4psv6 - sv5psv7, 0.0);
            var r6 = new Complex(sv0psv2 - sv1psv3, sv4psv6 - sv5psv7);

            var p11 = new Complex(sv0msv2, sv3 - sv1);
            var p13 = new Complex(sv0msv2, sv1 - sv3);

            const double c = 0.70710678118654757;

            var ep2k1 = new Complex(
                c * (sv7msv5 + sv4msv6),
                c * (sv7msv5 - sv4msv6));
            var ep2k3 = new Complex(
                c * (sv5 - sv7 - sv4msv6),
                c * (sv7msv5 - sv4msv6));

            return new Complex[8]
            {
                r0,
                p11 + ep2k1,
                r2,
                p13 + ep2k3,
                r4,
                p11 - ep2k1,
                r6,
                p13 - ep2k3
            };
        }

        Complex[] Fourier4(double[] signal, int ofs, int stride)
        {
            double sv0 = signal[ofs];
            double sv1 = signal[stride + ofs];
            double sv2 = signal[stride * 2 + ofs];
            double sv3 = signal[stride * 3 + ofs];
            double sv0psv2 = sv0 + sv2;
            double sv0msv2 = sv0 - sv2;
            double sv1psv3 = sv1 + sv3;
            return new Complex[4]
            {
                new Complex(sv0psv2 + sv1psv3, 0.0),
                new Complex(sv0msv2, sv3 - sv1),
                new Complex(sv0psv2 - sv1psv3, 0.0),
                new Complex(sv0msv2, sv1 - sv3)
            };
        }

        Complex[] Fourier2(double[] signal, int ofs, int stride)
        {
            double sv0 = signal[ofs];
            double sv1 = signal[stride + ofs];
            return new Complex[2]
            {
                new Complex(sv0 + sv1, 0.0),
                new Complex(sv0 - sv1, 0.0)
            };
        }
    }
}
