// SVG 가져오기에서 PNG 가져오기로 변경
import logo from '@/assets/images/flowise_white.png' // .svg에서 .png로 변경
import logoDark from '@/assets/images/flowise_dark.png' // .svg에서 .png로 변경
import { useSelector } from 'react-redux'

// ==============================|| LOGO ||============================== //
const Logo = () => {
    const customization = useSelector((state) => state.customization)
    return (
        <div style={{ alignItems: 'center', display: 'flex', flexDirection: 'row', marginLeft: '10px' }}>
            <img
                style={{ objectFit: 'contain', height: 'auto', width: 150 }}
                src={customization.isDarkMode ? logoDark : logo}
                alt='Linkbricks Horizon-AI'
            />
        </div>
    )
}
export default Logo
