using System;
using System.Linq;
using System.Windows.Media;
using System.Windows.Shapes;
using System.Windows.Controls;
using System.Windows.Threading;

namespace SoundReceiver
{
    class Diagram
    {
        Canvas m_C;

        public Diagram(Canvas C)
        {
            m_C = C;
        }

        public void Fill(double[] Data, double L1, double L2, int Count)
        {
            m_C.Dispatcher.Invoke(
                DispatcherPriority.Normal,
                new Action(() => 
                {
                    m_C.Children.Clear();

                    double DL1 = L2 - L1;
                    double DL2 = DL1 / Count + 0.0001;

                    int[] CollectA = new int[Count];
                    foreach (double D in Data)
                        if (D >= L1)
                            if (D <= L2)
                                CollectA[(int)((D - L1) / DL2)]++;

                    int Max = CollectA.Max();
                    double[] CollectAN = new double[Count];
                    if (Max != 0)
                        for (int i = 0; i < Count; i++)
                            CollectAN[i] = (double)CollectA[i] / Max;

                    SolidColorBrush B1 = new SolidColorBrush(Color.FromRgb(0xCC, 0xDD, 0xFF));
                    SolidColorBrush B2 = new SolidColorBrush(Color.FromRgb(0x88, 0xAA, 0xFF));
                    for (int i = 0; i < Count; i++)
                    {
                        Rectangle R = new Rectangle();
                        R.Opacity = 0.9;
                        R.Width = 8;
                        R.ToolTip = string.Format("{0} bit(s)", CollectA[i]);
                        R.Height = CollectAN[i] * 100.0;
                        R.Fill = B1;
                        R.Stroke = B2;
                        R.SetValue(Canvas.LeftProperty, i * 10.0 + 1.0);
                        R.SetValue(Canvas.TopProperty, 100.0 - R.Height);

                        m_C.Children.Add(R);
                    }
                }));
        }
    }
}
