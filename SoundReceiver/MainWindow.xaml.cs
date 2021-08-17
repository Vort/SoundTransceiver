using System;
using System.Text;
using System.Windows;
using System.Collections.Generic;
using System.Threading;
using System.Windows.Threading;
using NAudio.Wave;

namespace SoundReceiver
{
    public partial class MainWindow : Window
    {
        List<short> buf;
        WaveIn waveIn;

        Detector detector;

        Diagram bitLevelsDiagram;

        public MainWindow()
        {
            InitializeComponent();

            bitLevelsDiagram = new Diagram(Canvas1);
            detector = new Detector();

            waveIn = new WaveIn();
            waveIn.WaveFormat = new WaveFormat(44100, 1);
            waveIn.DataAvailable += waveIn_DataAvailable;
            waveIn.RecordingStopped += waveIn_RecordingStopped;
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
                waveIn.StartRecording();

                FreqSlider.IsEnabled = false;
                BitrateSlider.IsEnabled = false;
                StartButton.IsEnabled = false;
                StopButton.IsEnabled = true;
                buf = new List<short>();
            }
            catch
            {
            }
        }

        void StopButton_Click(object sender, RoutedEventArgs e)
        {
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
                StringBuilder sb = new StringBuilder();
                byte[] data = detector.Detect(
                    buf.ToArray(), bitLevels, out snr, out mer);
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
                    SNRTextBlock.Text = snr == null ? "—" : $"{snr,2} dB";
                    MERTextBlock.Text = mer == null ? "—" : $"{mer,2} dB";
                    MessageTextBox.Text = resultStr;

                    FreqSlider.IsEnabled = true;
                    BitrateSlider.IsEnabled = true;
                    StartButton.IsEnabled = true;
                }));
        }
    }
}
