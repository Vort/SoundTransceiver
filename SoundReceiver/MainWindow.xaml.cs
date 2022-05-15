using NAudio;
using NAudio.Wave;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Windows;
using System.Windows.Input;
using System.Windows.Threading;

namespace SoundReceiver
{
    public partial class MainWindow : Window
    {
        bool createWavFile;

        List<short> buf;
        WaveIn waveIn;

        Detector detector;

        Diagram bitLevelsDiagram;

        public MainWindow()
        {
            InitializeComponent();

            buf = new List<short>();
            bitLevelsDiagram = new Diagram(Canvas1);
            detector = new Detector();
        }

        private void waveIn_RecordingStopped(object sender, StoppedEventArgs e)
        {
            new Thread(ProcessSignal).Start();
        }

        private void waveIn_DataAvailable(object sender, WaveInEventArgs e)
        {
            for (int i = 0; i < e.BytesRecorded / 2; i++)
                buf.Add(BitConverter.ToInt16(e.Buffer, i * 2));
        }

        void UpdateParamInfo()
        {
            if (FreqTextBlock == null)
                return;

            detector.carrierFreq = FreqSlider.Value;
            FreqTextBlock.Text = detector.carrierFreq.ToString() + " Hz";
            detector.bitrate = BitrateSlider.Value;
            BitrateTextBlock.Text = detector.bitrate.ToString() + " bps";
        }

        private void Slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            UpdateParamInfo();
        }

        void StartButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                buf.Clear();

                // WaveIn needs to be reinitialized each time:
                // https://github.com/naudio/NAudio/issues/49#issuecomment-280446686
                waveIn?.Dispose();
                waveIn = new WaveIn();
                waveIn.WaveFormat = new WaveFormat(44100, 1);
                waveIn.DataAvailable += waveIn_DataAvailable;
                waveIn.RecordingStopped += waveIn_RecordingStopped;
                waveIn.StartRecording();

                FreqSlider.IsEnabled = false;
                BitrateSlider.IsEnabled = false;
                StartButton.IsEnabled = false;
                StopButton.IsEnabled = true;
            }
            catch (MmException ex)
            {
                if (ex.Result == MmResult.BadDeviceId)
                    MessageTextBox.Text = "[Recording device is not found]";
                else
                    MessageTextBox.Text = $"[Recording error: {ex.Result}]";
            }
        }

        void StopButton_Click(object sender, RoutedEventArgs e)
        {
            createWavFile = Keyboard.Modifiers.HasFlag(ModifierKeys.Control);

            waveIn.StopRecording();
            StopButton.IsEnabled = false;
        }

        void ProcessSignal()
        {
            string resultStr = "";
            double? snr = null;
            double? mer = null;
            var bitLevels = new List<double>();
            try
            {
                short[] bufA = buf.ToArray();
                if (createWavFile)
                {
                    string ts = DateTime.Now.ToString("yyyyMMddHHmmss");
                    Detector.SaveWav(
                        $"r_{ts}_{detector.carrierFreq}_{detector.bitrate}.wav", bufA);
                }
                byte[] data = detector.Detect(bufA, bitLevels, out snr, out mer);

                StringBuilder sb = new StringBuilder();
                string unfiltered = Encoding.UTF8.GetString(data);
                foreach (char c in unfiltered)
                    sb.Append(c < ' ' ? '�' : c);
                resultStr = sb.ToString();
            }
            catch (SignalException e)
            {
                resultStr = string.Format("[{0}]", e.Message);
            }
            bitLevelsDiagram.Fill(bitLevels.ToArray(), 0.001, 2.0, 39);

            Dispatcher.Invoke(
                DispatcherPriority.Normal,
                new Action(() =>
                {
                    SNRTextBlock.Text = snr == null ? "—" : $"{Convert.ToInt32(snr),2} dB";
                    MERTextBlock.Text = mer == null ? "—" : $"{Convert.ToInt32(mer),2} dB";
                    MessageTextBox.Text = resultStr;

                    FreqSlider.IsEnabled = true;
                    BitrateSlider.IsEnabled = true;
                    StartButton.IsEnabled = true;
                }));
        }
    }
}
