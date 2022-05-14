using System;
using System.IO;
using System.Linq;
using System.Media;
using System.Text;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Threading;

namespace SoundTransmitter
{
    public partial class MainWindow : Window
    {
        const int sampleRate = 44100;

        double bitrate = 100;
        double carrierFreq = 1000;

        sbyte[] equalizerTrainSequence = new sbyte[]
        {
            1, 1, 1, 1, 1, -1, 1, -1,
            -1, -1, 1, -1, 1, 1, 1, 1,
            1, -1, 1, -1, -1, 1, 1, -1,
            -1, -1, -1, 1, -1, -1, -1, -1
        };
        int sendRrcBitCount = 64;
        double rrcBeta = 0.8;

        int payloadSizeLsbSize = 3;

        string message;


        public MainWindow()
        {
            InitializeComponent();
        }

        void UpdateParamInfo()
        {
            if (FreqTextBlock == null)
                return;

            carrierFreq = FreqSlider.Value;
            FreqTextBlock.Text = carrierFreq.ToString() + " Hz";
            bitrate = BitrateSlider.Value;
            BitrateTextBlock.Text = bitrate.ToString() + " bps";

            SendButton.IsEnabled = MessageTextBox.Text.Length != 0;
        }

        private void Slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            UpdateParamInfo();
        }

        private void MessageTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (SendButton != null)
                SendButton.IsEnabled = MessageTextBox.Text.Length != 0;
        }

        private void SendButton_Click(object sender, RoutedEventArgs e)
        {
            FreqSlider.IsEnabled = false;
            BitrateSlider.IsEnabled = false;
            MessageTextBox.IsEnabled = false;
            SendButton.IsEnabled = false;

            message = MessageTextBox.Text;
            new Thread(MakeSignal).Start();
        }

        void MakeSignal()
        {
            var data = Encoding.UTF8.GetBytes(message);

            sbyte[] symbols = MakePacketSymbols(data);
            double[] digitalSignal = Mul(MakePulsesShape(symbols,
                sampleRate, bitrate, rrcBeta, sendRrcBitCount), 0.67);
            double[] aModSignal = AModulation(digitalSignal, carrierFreq);
            // Delay is needed to prevent mouse and keyboard noises from mixing with signal
            double[] silence = MakeSilence(0.1);
            double[] result = silence.Concat(aModSignal).ToArray();

            MemoryStream wav = MakeWav(SignalDtoS(result), 2);

            SendButton.Dispatcher.Invoke(
                DispatcherPriority.Normal,
                new Action(() => { SendButton.Content = "Play!"; }));

            var Player = new SoundPlayer(wav);
            Player.PlaySync();

            FreqSlider.Dispatcher.Invoke(
                DispatcherPriority.Normal,
                new Action(() => { FreqSlider.IsEnabled = true; }));
            BitrateSlider.Dispatcher.Invoke(
                DispatcherPriority.Normal,
                new Action(() => { BitrateSlider.IsEnabled = true; }));
            MessageTextBox.Dispatcher.Invoke(
                DispatcherPriority.Normal,
                new Action(() => { MessageTextBox.IsEnabled = true; }));
            SendButton.Dispatcher.Invoke(
                DispatcherPriority.Normal,
                new Action(() => { SendButton.IsEnabled = true; }));

            SendButton.Dispatcher.Invoke(
                DispatcherPriority.Normal,
                new Action(() => { SendButton.Content = "Send"; }));
        }

        double[] MakeSilence(double T)
        {
            return new double[(int)(T * sampleRate)];
        }

        sbyte[] MakePacketSymbols(byte[] payload)
        {
            sbyte[] payloadSizeLsb = new sbyte[payloadSizeLsbSize];
            for (int i = 0; i < payloadSizeLsbSize; i++)
            {
                int mask = 1 << (payloadSizeLsbSize - i - 1);
                payloadSizeLsb[i] = (sbyte)((payload.Length & mask) == 0 ? -1 : 1);
            }
            sbyte[] symbols = new sbyte[payload.Length * 8];
            for (int i = 0; i < payload.Length; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    bool bit = false;
                    if ((payload[i] & (1 << (7 - j))) != 0)
                        bit = true;
                    symbols[i * 8 + j] = (sbyte)(bit ? 1 : -1);
                }
            }

            return equalizerTrainSequence.Concat(payloadSizeLsb).Concat(symbols).ToArray();
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

        double[] Mul(double[] signal, double value)
        {
            double[] r = new double[signal.Length];

            for (int i = 0; i < signal.Length; i++)
                r[i] = signal[i] * value;

            return r;
        }
        
        double[] AModulation(double[] signal, double freq)
        {
            double[] modSignal = new double[signal.Length];
            for (int i = 0; i < signal.Length; i++)
                modSignal[i] = signal[i] * Math.Cos(2 * Math.PI * (freq * i / sampleRate));
            return modSignal;
        }



        short[] SignalDtoS(double[] Signal)
        {
            short[] SignalS = new short[Signal.Length];
            for (int i = 0; i < Signal.Length; i++)
                SignalS[i] = (short)(Signal[i] * 32767.0);
            return SignalS;
        }

        MemoryStream MakeWav(short[] signal, int channelCount, int sampleRate = sampleRate)
        {
            var ms = new MemoryStream();
            ms.Write(new byte[] { 0x52, 0x49, 0x46, 0x46 }, 0, 4); // "RIFF"
            ms.Write(BitConverter.GetBytes((uint)(36 + signal.Length * 2 * channelCount)), 0, 4);
            ms.Write(new byte[] { 0x57, 0x41, 0x56, 0x45 }, 0, 4); // "WAVE"
            ms.Write(new byte[] { 0x66, 0x6D, 0x74, 0x20 }, 0, 4); // "fmt"
            ms.Write(BitConverter.GetBytes((uint)(16)), 0, 4);
            ms.Write(BitConverter.GetBytes((ushort)(1)), 0, 2);
            ms.Write(BitConverter.GetBytes((ushort)(channelCount)), 0, 2);
            ms.Write(BitConverter.GetBytes((uint)(sampleRate)), 0, 4); // Hz
            ms.Write(BitConverter.GetBytes((uint)(sampleRate * 2 * channelCount)), 0, 4);
            ms.Write(BitConverter.GetBytes((ushort)(2 * channelCount)), 0, 2);
            ms.Write(BitConverter.GetBytes((ushort)(16)), 0, 2); // bps
            ms.Write(new byte[] { 0x64, 0x61, 0x74, 0x61 }, 0, 4); // "data"
            ms.Write(BitConverter.GetBytes((uint)(signal.Length * 2 * channelCount)), 0, 4);
            foreach (short v in signal)
            {
                ms.Write(BitConverter.GetBytes(v), 0, 2);
                for (int i = 0; i < channelCount - 1; i++)
                    ms.Write(BitConverter.GetBytes((ushort)0), 0, 2);
            }
            ms.Position = 0;
            return ms;
        }
    }
}
