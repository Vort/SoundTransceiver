using System;
using System.Windows;
using System.Text;
using System.Linq;
using System.IO;
using System.Windows.Threading;
using System.Threading;

namespace SoundTransmitter
{
    public partial class MainWindow : Window
    {
        double m_Bitrate = 100;
        double m_SineFreq1 = 800;
        double m_SineFreq2 = 1200;

        const int m_SampleRate = 44100;

        string m_Message;


        public MainWindow()
        {
            InitializeComponent();
        }

        void UpdateParamInfo()
        {
            if (Freq1TextBlock == null)
                return;

            m_SineFreq1 = Freq1Slider.Value;
            Freq1TextBlock.Text = m_SineFreq1.ToString() + " Hz";
            m_SineFreq2 = Freq2Slider.Value;
            Freq2TextBlock.Text = m_SineFreq2.ToString() + " Hz";
            m_Bitrate = BitrateSlider.Value;
            BitrateTextBlock.Text = m_Bitrate.ToString() + " bps";

            SendButton.IsEnabled = (m_SineFreq1 != m_SineFreq2) && (MessageTextBox.Text.Length != 0);
        }

        private void Slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            UpdateParamInfo();
        }

        private void MessageTextBox_TextChanged(object sender, System.Windows.Controls.TextChangedEventArgs e)
        {
            if (SendButton != null)
                SendButton.IsEnabled = (m_SineFreq1 != m_SineFreq2) && (MessageTextBox.Text.Length != 0);
        }

        private void SendButton_Click(object sender, RoutedEventArgs e)
        {
            Freq1Slider.IsEnabled = false;
            Freq2Slider.IsEnabled = false;
            BitrateSlider.IsEnabled = false;
            MessageTextBox.IsEnabled = false;
            SendButton.IsEnabled = false;

            m_Message = MessageTextBox.Text;
            new Thread(MakeSignal).Start();
        }

        void MakeSignal()
        {
            byte[] MsgS = Encoding.UTF8.GetBytes(m_Message);
            byte[] BitStream = Encode(MsgS);
            double[] Gaussian = MakeGaussian(1 / m_Bitrate);
            double[] DigitalSignal = MakeDigitalSignal(m_Bitrate, BitStream);
            double[] AmpSignal1 = MakeAmpSignal(m_Bitrate, DigitalSignal.Length);
            double[] AmpSignal2 = Normalize(Convolution2(Gaussian, AmpSignal1));
            double[] DigitalSignal3 = Normalize(Convolution2(Gaussian, DigitalSignal));
            double[] FModSignal = FModulation(DigitalSignal3, m_SineFreq1, m_SineFreq2);
            double[] AFModSignal = AModulation(FModSignal, AmpSignal2);
            double[] Silence = MakeSilence(0.1);
            double[] Result = Silence.Concat(AFModSignal).Concat(Silence).ToArray();

            /*
            SaveWav("1.wav", SignalDtoS(Gaussian));
            SaveWav("2.wav", SignalDtoS(DigitalSignal));
            SaveWav("3.wav", SignalDtoS(AmpSignal1));
            SaveWav("4.wav", SignalDtoS(AmpSignal2));
            SaveWav("5.wav", SignalDtoS(DigitalSignal3));
            SaveWav("6.wav", SignalDtoS(FModSignal));
            SaveWav("7.wav", SignalDtoS(AFModSignal));
            */

            string outName = string.Format("TS_{0}_{1}_{2}.wav",
                m_SineFreq1, m_SineFreq2, m_Bitrate);
            SaveWav(outName, SignalDtoS(Result));

            for (int i = 0; i < 3; i++)
            {
                SendButton.Dispatcher.Invoke(
                    DispatcherPriority.Normal,
                    new Action(() => { SendButton.Content = (3 - i); }));
                System.Threading.Thread.Sleep(500);
            }
            SendButton.Dispatcher.Invoke(
                DispatcherPriority.Normal,
                new Action(() => { SendButton.Content = "Play!"; }));


            var Player = new System.Media.SoundPlayer();
            Player.SoundLocation = outName;
            Player.PlaySync();

            File.Delete(outName);


            Freq1Slider.Dispatcher.Invoke(
                DispatcherPriority.Normal,
                new Action(() => { Freq1Slider.IsEnabled = true; }));
            Freq2Slider.Dispatcher.Invoke(
                DispatcherPriority.Normal,
                new Action(() => { Freq2Slider.IsEnabled = true; }));
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
            return new double[(int)(T * m_SampleRate)];
        }

        byte[] Encode(byte[] Message)
        {
            byte[] BitStream = new byte[Message.Length * 8];
            for (int i = 0; i < Message.Length; i++)
            {
                for (int j = 0; j < 8; j++)
                {
                    bool Bit = false;
                    if ((Message[i] & (1 << (7 - j))) != 0)
                        Bit = true;
                    BitStream[i * 8 + j] = (byte)(Bit ? 1 : 0);
                }
            }
            return new byte[] { 0, 0, 0, 0, 1, 0 }.Concat(BitStream).Concat(new byte[] { 0, 0, 0, 0 }).ToArray();
        }

        double[] MakeGaussian(double Duration)
        {
            int SignalLen = (int)(Duration * m_SampleRate);

            double A = Math.PI * 2;
            double[] Gaussian = new double[SignalLen];

            for (int i = 0; i < SignalLen; i++)
            {
                double X = (2 * i / (double)SignalLen - 1);
                Gaussian[i] = Math.Exp(-A * X * X);
            }
            return Gaussian;
        }

        double[] MakeAmpSignal(double Bitrate, int SignalLen)
        {
            double BitLen = m_SampleRate / Bitrate;
            double[] AmpSignal = new double[SignalLen];
            for (int i = 0; i < SignalLen; i++)
                AmpSignal[i] = ((i > BitLen / 2) && (i < SignalLen - BitLen / 2)) ? 1 : -1;
            return AmpSignal;
        }

        double[] MakeDigitalSignal(double Bitrate, byte[] Bits)
        {
            int SignalLen = (int)((Bits.Length / Bitrate) * m_SampleRate);

            double[] DigitalSignal = new double[SignalLen];

            for (int i = 0; i < SignalLen; i++)
            {
                int BitIndex = (int)(Bitrate * i / m_SampleRate);
                DigitalSignal[i] = (Bits[BitIndex] == 1) ? 1 : -1;
            }

            return DigitalSignal;
        }

        double[] PadSignal(double Bitrate, double BitCount, double[] Signal)
        {
            double[] Pad = new double[(int)(BitCount * m_SampleRate / Bitrate)];
            return Pad.Concat(Signal).Concat(Pad).ToArray();
        }

        double[] Normalize(double[] Signal)
        {
            double Min = double.MaxValue;
            double Max = double.MinValue;
            foreach (double D in Signal)
            {
                if (D < Min)
                    Min = D;
                if (D > Max)
                    Max = D;
            }

            double MaxAbs = Math.Max(Math.Abs(Min), Math.Abs(Max));

            double[] NormSignal = new double[Signal.Length];
            for (int i = 0; i < Signal.Length; i++)
                NormSignal[i] = (Signal[i] / MaxAbs);
            return NormSignal;
        }

        double[] Convolution2(double[] W1, double[] W2)
        {
            int padLen = W1.Length / 2;
            double[] pad1 = new double[padLen];
            double[] pad2 = new double[padLen];
            for (int i = 0; i < padLen; i++)
                pad1[i] = W2[0];
            for (int i = 0; i < padLen; i++)
                pad2[i] = W2[W2.Length - 1];
            return Convolution(W1, pad1.Concat(W2).Concat(pad2).ToArray());
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
                R[i] = Sum / W1.Length;
            }
            return R;
        }

        double[] AModulation(double[] DataSignal, double[] AmpSignal)
        {
            double[] Signal = new double[Math.Min(DataSignal.Length, AmpSignal.Length)];

            for (int i = 0; i < Signal.Length; i++)
                Signal[i] = DataSignal[i] * (AmpSignal[i] * 0.5 + 0.5);

            return Signal;
        }

        double[] FModulation(double[] Signal, double Freq1, double Freq2)
        {
            double CF = (Freq2 + Freq1) / 2;
            double D = (Freq2 - Freq1) / 2;
            double[] ModSignal = new double[Signal.Length];

            double Phase = 0;
            for (int i = 0; i < Signal.Length; i++)
            {
                Phase += 2 * Math.PI * (CF + D * Signal[i]) / (double)m_SampleRate;
                ModSignal[i] = Math.Sin(Phase);
            }
            return ModSignal;
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
    }
}
