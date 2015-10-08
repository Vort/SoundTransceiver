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
        List<short> m_Buf;
        WaveIn waveIn;

        Detector m_D;

        Diagram m_DG1;
        Diagram m_DG2;

        public MainWindow()
        {
            InitializeComponent();

            m_DG1 = new Diagram(Canvas1);
            m_DG2 = new Diagram(Canvas2);
            m_D = new Detector(m_DG1, m_DG2);

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
                m_Buf.Add(BitConverter.ToInt16(e.Buffer, i * 2));
        }

        void UpdateParamInfo()
        {
            if (Freq1TextBlock == null)
                return;

            m_D.sineFreq1 = Freq1Slider.Value;
            Freq1TextBlock.Text = m_D.sineFreq1.ToString() + " Hz";
            m_D.sineFreq2 = Freq2Slider.Value;
            Freq2TextBlock.Text = m_D.sineFreq2.ToString() + " Hz";
            m_D.bitrate = BitrateSlider.Value;
            BitrateTextBlock.Text = m_D.bitrate.ToString() + " bps";

            StartButton.IsEnabled = m_D.sineFreq1 != m_D.sineFreq2;
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

                Freq1Slider.IsEnabled = false;
                Freq2Slider.IsEnabled = false;
                BitrateSlider.IsEnabled = false;
                StartButton.IsEnabled = false;
                StopButton.IsEnabled = true;
                MessageTextBox.Text = "";
                SNRTextBlock.Text = "—";
                m_Buf = new List<short>();
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
            string snr = "—";
            string resultStr = "";
            try
            {
                StringBuilder sb = new StringBuilder();
                byte[] data = m_D.Detect(m_Buf.ToArray(), ref snr);
                string unfiltered = Encoding.UTF8.GetString(data);
                foreach (char c in unfiltered)
                    sb.Append(c < ' ' ? '�' : c);
                resultStr = sb.ToString();
            }
            catch (SignalException e)
            {
                resultStr = string.Format("[{0}]", e.Message);
            }

            Dispatcher.Invoke(
                DispatcherPriority.Normal,
                new Action(() =>
                {
                    SNRTextBlock.Text = snr;
                    MessageTextBox.Text = resultStr;

                    Freq1Slider.IsEnabled = true;
                    Freq2Slider.IsEnabled = true;
                    BitrateSlider.IsEnabled = true;
                    StartButton.IsEnabled = true;
                }));
        }
    }
}
