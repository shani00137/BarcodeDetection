using System;
using System.Collections.Generic;
using System.Linq;
using System.Media;
using System.Text;
using System.Threading.Tasks;

namespace BarcodeDetection.Utilities
{
    public class MediaBeep
    {
        public static void PlayRobotBeep()
        {
            try
            {
                var player = new SoundPlayer("Asset/beep.wav");
                player.Play();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Sound error: " + ex.Message);
            }
        }
    }
}
