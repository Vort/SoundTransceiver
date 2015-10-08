using System;
using System.Collections.Generic;

namespace SoundReceiver
{
    class BitContainer
    {
        byte m_CurByte = 0;
        int m_CurBitShift = 0;
        List<byte> m_Data = new List<byte>();

        public byte[] ExtractData()
        {
            byte[] R = m_Data.ToArray();
            m_Data = new List<byte>();
            return R;
        }

        public void Align()
        {
            for (; ; )
            {
                if (m_CurBitShift == 0)
                    break;

                Add(false);
            }
        }

        public void Add(bool B)
        {
            if (B)
                m_CurByte = (byte)(m_CurByte | (1 << (7 - m_CurBitShift)));

            m_CurBitShift++;
            if (m_CurBitShift == 8)
            {
                m_Data.Add(m_CurByte);
                m_CurByte = 0;
                m_CurBitShift = 0;
            }
        }

        public void Add(uint B, int Count)
        {
            for (int i = 0; i < Count; i++)
            {
                Add((B & (1 << (Count - i - 1))) != 0);
            }
        }
    }
}
